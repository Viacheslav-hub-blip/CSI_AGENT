import asyncio
import io
import json
import os
import threading
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional
import nest_asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from data_normalization_subgraph import normalization_service
from langgraph_agent import CSIAgent
from load_data import get_data
from tools import DATAFRAMES_DIR, main_react_agent_sandbox

nest_asyncio.apply()
COMPUTE_TOOL_URL = "http://127.0.0.1:8200"
HOST = "127.0.0.1"
PORT = 8113
MAX_CONTEXT_MESSAGES = 7
SESSION_CLEANUP_THRESHOLD = 100
SESSION_MAX_AGE_SECONDS = 3600
STEP_INPUT_PREVIEW_LIMIT = 500
MAX_THREAD_NOTIFICATIONS = 200
USER_MESSAGE_PREFIX = " ##USER##: "
AI_MESSAGE_PREFIX = " ##AI AGENT##:"
UPLOADS_SUBDIR = "uploads"
EXTRA_DATAFRAME_PREFIX = "extra_"
COMPUTE_DATAFRAME_PREFIX = "computed_"
COMPUTE_CALLBACK_PATH = "/compute-jobs/callback"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ComputeJobCallback(BaseModel):
    event_type: str
    thread_id: str
    job_id: str
    message: str = ""
    processed_rows: int | None = None
    total_rows: int | None = None
    result_url: str = ""
    row_count: int | None = None
    column_count: int | None = None
    error: str = ""


print("Loading data...")
os.makedirs(DATAFRAMES_DIR, exist_ok=True)
source_path = os.path.join(DATAFRAMES_DIR, "source_dataframe.pkl")
current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")


def _load_or_create_source_dataframe() -> pd.DataFrame:
    if os.path.exists(source_path):
        return pd.read_pickle(source_path).copy()
    loaded_df = get_data()
    loaded_df.to_pickle(source_path)
    return loaded_df.copy()


source_df = _load_or_create_source_dataframe()
df = source_df.copy()
if not os.path.exists(current_path):
    source_df.to_pickle(current_path)
print(f"Data loaded: shape={df.shape}")
_session_messages: Dict[str, List[str]] = {}
_session_flags: Dict[str, Dict[str, Any]] = {}
_thread_notifications: Dict[str, List[Dict[str, Any]]] = {}
_compute_jobs_lock = threading.Lock()


def clear_figures_from_sandbox(sandbox) -> None:
    figure_names = [
        name for name, value in sandbox.globals.items() if isinstance(value, go.Figure)
    ]
    for name in figure_names:
        del sandbox.globals[name]


def extract_figures_from_sandbox(sandbox) -> List[Dict[str, str]]:
    figures: List[Dict[str, str]] = []
    seen_ids: set[int] = set()
    for name, value in sandbox.globals.items():
        if name.startswith("_") or not isinstance(value, go.Figure):
            continue
        object_id = id(value)
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        try:
            figures.append({"name": name, "figure_json": pio.to_json(value)})
        except Exception as exc:
            print(f"Could not serialize figure {name}: {exc}")
    return figures


def _get_or_create_session(thread_id: str) -> tuple[List[str], Dict[str, Any]]:
    if thread_id not in _session_messages:
        _session_messages[thread_id] = []
        _session_flags[thread_id] = {
            "active_calculate_chain": False,
            "active_normalization_chain": False,
            "current_step_in_calculate_chain": 0,
            "last_activity": time.time(),
        }
    _session_flags[thread_id]["last_activity"] = time.time()
    return _session_messages[thread_id], _session_flags[thread_id]


def _cleanup_old_sessions(max_age_seconds: int = SESSION_MAX_AGE_SECONDS) -> None:
    now = time.time()
    expired_thread_ids = [
        thread_id
        for thread_id, flags in _session_flags.items()
        if now - flags.get("last_activity", 0) > max_age_seconds
    ]
    for thread_id in expired_thread_ids:
        _session_messages.pop(thread_id, None)
        _session_flags.pop(thread_id, None)
        normalization_service.clear_session(f"normalization::{thread_id}")
    if expired_thread_ids:
        print(f"Removed {len(expired_thread_ids)} stale sessions")


def _cleanup_sessions_if_needed() -> None:
    if len(_session_messages) > SESSION_CLEANUP_THRESHOLD:
        _cleanup_old_sessions()


def _make_safe_updates(updates: Optional[dict]) -> Dict[str, Any]:
    if updates is None:
        return {}
    safe_updates: Dict[str, Any] = {}
    for key, value in updates.items():
        if isinstance(value, pd.DataFrame):
            safe_updates[key] = f"DataFrame shape={value.shape}"
        elif isinstance(value, (str, int, float, bool)):
            safe_updates[key] = value
        else:
            safe_updates[key] = type(value).__name__
    return safe_updates


def _get_current_df() -> pd.DataFrame:
    if os.path.exists(current_path):
        return pd.read_pickle(current_path)
    return df


def _append_user_message(messages: List[str], message: str) -> None:
    messages.append(USER_MESSAGE_PREFIX + message)


def _append_ai_message(messages: List[str], message: str) -> None:
    messages.append(AI_MESSAGE_PREFIX + message)


def _build_agent_state(
    session_messages: List[str],
    flags: Dict[str, Any],
    current_dataframe: pd.DataFrame,
    start_time: float,
) -> Dict[str, Any]:
    return {
        "user_input": " ".join(session_messages[-MAX_CONTEXT_MESSAGES:]),
        "active_calculate_chain": flags["active_calculate_chain"],
        "active_normalization_chain": flags.get("active_normalization_chain", False),
        "current_step_in_calculate_chain": flags["current_step_in_calculate_chain"],
        "start_time": start_time,
        "last_change_df": current_dataframe,
    }


def _prepare_agent_run(
    request: ChatRequest,
) -> tuple[List[str], Dict[str, Any], CSIAgent, pd.DataFrame, float]:
    session_messages, flags = _get_or_create_session(request.thread_id)
    _append_user_message(session_messages, request.message)
    current_dataframe = _get_current_df()
    clear_figures_from_sandbox(main_react_agent_sandbox)
    agent = CSIAgent(
        current_dataframe,
        COMPUTE_TOOL_URL,
        thread_id=request.thread_id,
        compute_callback_url=_build_compute_callback_url(),
        compute_callback_thread_id=request.thread_id,
    )
    return session_messages, flags, agent, current_dataframe, time.time()


def _build_compute_dataframe_name(job_id: str) -> str:
    return f"{COMPUTE_DATAFRAME_PREFIX}{job_id[:8]}.pkl"


def _build_compute_callback_url() -> str:
    return f"http://{HOST}:{PORT}{COMPUTE_CALLBACK_PATH}"


def _push_thread_notification(
    thread_id: str,
    status: str,
    message: str,
    job_id: str,
    dataframe_name: str = "",
    processed_rows: int | None = None,
    total_rows: int | None = None,
) -> None:
    payload = {
        "id": f"{job_id}:{datetime.now().timestamp()}",
        "status": status,
        "message": message,
        "job_id": job_id,
        "dataframe_name": dataframe_name,
        "processed_rows": processed_rows,
        "total_rows": total_rows,
        "timestamp": datetime.now().isoformat(),
    }
    with _compute_jobs_lock:
        notifications = _thread_notifications.setdefault(thread_id, [])
        notifications.append(payload)
        if len(notifications) > MAX_THREAD_NOTIFICATIONS:
            del notifications[:-MAX_THREAD_NOTIFICATIONS]


def _save_computed_dataframe(job_id: str, result_url: str) -> tuple[str, list[int]]:
    response = requests.get(f"{COMPUTE_TOOL_URL}{result_url}", timeout=60)
    response.raise_for_status()
    dataframe = pd.read_pickle(io.BytesIO(response.content))
    dataframe_name = _build_compute_dataframe_name(job_id)
    dataframe_path = os.path.join(DATAFRAMES_DIR, dataframe_name)
    dataframe.to_pickle(dataframe_path)
    try:
        dataframe.to_excel(dataframe_path.replace(".pkl", ".xlsx"), index=False)
    except Exception as exc:
        print(f"Could not export compute result to xlsx: {exc}")
    return dataframe_name, list(dataframe.shape)


def _build_compute_progress_message(
    job_id: str,
    processed_rows: int | None,
    total_rows: int | None,
    message: str,
) -> str:
    normalized_message = message.strip() or "Фоновое вычисление продолжается."
    if processed_rows is not None and total_rows is not None:
        return (
            f"{normalized_message} "
            f"Обработано {processed_rows} из {total_rows} строк. "
            f"Job: `{job_id}`."
        )
    return f"{normalized_message} Job: `{job_id}`."


def _apply_agent_updates(
    flags: Dict[str, Any],
    updates: Optional[dict],
    final_answer: str,
) -> str:
    if not updates:
        return final_answer
    if "final_answer" in updates:
        final_answer = updates["final_answer"]
    if "active_calculate_chain" in updates:
        flags["active_calculate_chain"] = updates["active_calculate_chain"]
    if "active_normalization_chain" in updates:
        flags["active_normalization_chain"] = updates["active_normalization_chain"]
    if "current_step_in_calculate_chain" in updates:
        flags["current_step_in_calculate_chain"] = updates[
            "current_step_in_calculate_chain"
        ]
    return final_answer


def _iter_agent_updates(
    agent: CSIAgent,
    session_messages: List[str],
    flags: Dict[str, Any],
    current_dataframe: pd.DataFrame,
    start_time: float,
) -> Iterator[tuple[str, Optional[dict], Dict[str, Any], str]]:
    final_answer = ""
    state = _build_agent_state(session_messages, flags, current_dataframe, start_time)
    for chunk in agent.app.stream(state, stream_mode="updates"):
        for node_name, updates in chunk.items():
            safe_updates = _make_safe_updates(updates)
            final_answer = _apply_agent_updates(flags, updates, final_answer)
            yield node_name, updates, safe_updates, final_answer


def _run_agent_to_completion(
    agent: CSIAgent,
    session_messages: List[str],
    flags: Dict[str, Any],
    current_dataframe: pd.DataFrame,
    start_time: float,
) -> tuple[str, List[dict], List[Dict[str, str]]]:
    steps: List[dict] = []
    final_answer = ""
    for node_name, updates, safe_updates, final_answer in _iter_agent_updates(
        agent,
        session_messages,
        flags,
        current_dataframe,
        start_time,
    ):
        step_payload = _build_step_payload(node_name, safe_updates)
        if updates is not None:
            step_payload["updates"] = safe_updates
        steps.append(step_payload)
    figures = extract_figures_from_sandbox(main_react_agent_sandbox)
    _append_ai_message(session_messages, final_answer)
    _cleanup_sessions_if_needed()
    return final_answer, steps, figures


def _stream_agent_run_to_queue(
    loop: asyncio.AbstractEventLoop,
    queue: "asyncio.Queue[Dict[str, Any] | None]",
    agent: CSIAgent,
    session_messages: List[str],
    flags: Dict[str, Any],
    current_dataframe: pd.DataFrame,
    start_time: float,
) -> None:
    try:
        final_answer = ""
        for node_name, _updates, safe_updates, final_answer in _iter_agent_updates(
            agent,
            session_messages,
            flags,
            current_dataframe,
            start_time,
        ):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "step", **_build_step_payload(node_name, safe_updates)},
            )
        for figure in extract_figures_from_sandbox(main_react_agent_sandbox):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "figure",
                    "name": figure["name"],
                    "figure_json": figure["figure_json"],
                },
            )
        _append_ai_message(session_messages, final_answer)
        loop.call_soon_threadsafe(
            queue.put_nowait,
            _build_done_payload(final_answer, start_time),
        )
        _cleanup_sessions_if_needed()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        loop.call_soon_threadsafe(queue.put_nowait, _build_error_payload(exc))
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)


def _build_step_payload(node_name: str, safe_updates: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node": node_name,
        "name": node_name,
        "timestamp": datetime.now().isoformat(),
        "input": json.dumps(safe_updates, ensure_ascii=False, default=str)[
            :STEP_INPUT_PREVIEW_LIMIT
        ],
        "output": safe_updates.get(
            "final_answer", safe_updates.get("react_agent_answer", "")
        ),
    }


def _build_done_payload(final_answer: str, start_time: float) -> Dict[str, Any]:
    return {
        "type": "done",
        "output": final_answer,
        "duration_seconds": round(time.time() - start_time, 2),
    }


def _build_error_payload(error: Exception) -> Dict[str, str]:
    return {"type": "error", "output": f"Ошибка: {error}"}


def _serialize_sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _get_uploads_dir() -> str:
    uploads_dir = os.path.join(DATAFRAMES_DIR, UPLOADS_SUBDIR)
    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir


def _load_dataframe_from_bytes(contents: bytes, extension: str) -> pd.DataFrame:
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
    }
    reader = readers.get(extension)
    if reader is None:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {extension}. Поддерживаются: CSV, Excel, JSON, Pickle",
        )
    return reader(io.BytesIO(contents))


def _save_uploaded_source_file(thread_id: str, filename: str, contents: bytes) -> str:
    file_path = os.path.join(_get_uploads_dir(), f"{thread_id}_{filename}")
    with open(file_path, "wb") as file_obj:
        file_obj.write(contents)
    return file_path


def _save_uploaded_dataframe(
    thread_id: str, dataframe: pd.DataFrame, replace_data: bool
) -> None:
    if replace_data:
        dataframe.to_pickle(current_path)
        print(f"DataFrame replaced: shape={dataframe.shape}")
        return
    extra_path = os.path.join(
        DATAFRAMES_DIR, f"{EXTRA_DATAFRAME_PREFIX}{thread_id}.pkl"
    )
    dataframe.to_pickle(extra_path)
    print(f"Additional dataframe saved: shape={dataframe.shape}")


def _build_dataframe_info(
    name: str,
    dataframe_path: str,
    dataframe: pd.DataFrame,
    is_current: bool,
) -> Dict[str, Any]:
    return {
        "name": name,
        "path": dataframe_path,
        "shape": list(dataframe.shape),
        "columns": list(dataframe.columns),
        "is_current": is_current,
    }


app = FastAPI(
    title="CSI Agent API",
    description="API for dataframe analysis with a graph-based AI agent.",
    version="1.0.0",
)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):

    async def event_generator() -> AsyncGenerator[str, None]:
        session_messages, flags, agent, current_dataframe, start_time = (
            _prepare_agent_run(request)
        )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
        yield _serialize_sse_event(
            {
                "type": "step",
                "name": "request_started",
                "timestamp": datetime.now().isoformat(),
                "input": "",
                "output": "",
            }
        )
        worker = threading.Thread(
            target=_stream_agent_run_to_queue,
            args=(
                loop,
                queue,
                agent,
                session_messages,
                flags,
                current_dataframe,
                start_time,
            ),
            daemon=True,
        )
        worker.start()
        while True:
            payload = await queue.get()
            if payload is None:
                break
            yield _serialize_sse_event(payload)
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> dict:
    try:
        session_messages, flags, agent, current_dataframe, start_time = (
            _prepare_agent_run(request)
        )
        final_answer, steps, figures = await asyncio.to_thread(
            _run_agent_to_completion,
            agent,
            session_messages,
            flags,
            current_dataframe,
            start_time,
        )
        steps.append(
            {
                "node": "_meta",
                "name": "_meta",
                "input": "",
                "output": "",
                "duration_seconds": round(time.time() - start_time, 2),
            }
        )
        return {"output": final_answer, "steps": steps, "figures": figures}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {"output": f"Ошибка: {exc}", "steps": [], "figures": []}


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "dataframe_shape": list(df.shape),
        "active_sessions": len(_session_messages),
        "dataframes_dir": DATAFRAMES_DIR,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reset")
async def reset_session(request: ChatRequest) -> dict:
    _session_messages.pop(request.thread_id, None)
    _session_flags.pop(request.thread_id, None)
    normalization_service.clear_session(f"normalization::{request.thread_id}")
    source_df.copy().to_pickle(current_path)
    main_react_agent_sandbox.reset(keep_base=True)
    clear_figures_from_sandbox(main_react_agent_sandbox)
    return {
        "status": "reset",
        "thread_id": request.thread_id,
        "message": "Сессия сброшена, DataFrame восстановлен",
    }


@app.get("/sessions")
async def list_sessions() -> dict:
    sessions_info: Dict[str, Any] = {}
    for thread_id, messages in _session_messages.items():
        flags = _session_flags.get(thread_id, {})
        sessions_info[thread_id] = {
            "message_count": len(messages),
            "active_calculate_chain": flags.get("active_calculate_chain", False),
            "active_normalization_chain": flags.get(
                "active_normalization_chain", False
            ),
            "last_activity": flags.get("last_activity", 0),
        }
    return {"sessions": sessions_info}


@app.post(COMPUTE_CALLBACK_PATH)
async def compute_job_callback(payload: ComputeJobCallback) -> dict:
    event_type = payload.event_type.strip().lower()
    if event_type == "progress":
        _push_thread_notification(
            thread_id=payload.thread_id,
            status="progress",
            job_id=payload.job_id,
            processed_rows=payload.processed_rows,
            total_rows=payload.total_rows,
            message=_build_compute_progress_message(
                payload.job_id,
                payload.processed_rows,
                payload.total_rows,
                payload.message,
            ),
        )
        return {"status": "accepted"}
    if event_type == "result":
        if not payload.result_url:
            raise HTTPException(
                status_code=400, detail="result_url is required for result events"
            )
        dataframe_name, shape = await asyncio.to_thread(
            _save_computed_dataframe,
            payload.job_id,
            payload.result_url,
        )
        _push_thread_notification(
            thread_id=payload.thread_id,
            status="success",
            job_id=payload.job_id,
            dataframe_name=dataframe_name,
            message=(
                "Вычисление завершено. "
                f"Новый DataFrame сохранен как `{dataframe_name}` ({shape[0]}x{shape[1]}). "
                "Теперь его можно попросить показать или заменить им текущий DataFrame."
            ),
        )
        return {"status": "accepted"}
    if event_type == "error":
        error_message = (
            payload.error.strip()
            or payload.message.strip()
            or "Неизвестная ошибка compute_tool."
        )
        _push_thread_notification(
            thread_id=payload.thread_id,
            status="error",
            job_id=payload.job_id,
            message=f"Вычисление завершилось ошибкой для job `{payload.job_id}`: {error_message}",
        )
        return {"status": "accepted"}
    raise HTTPException(
        status_code=400, detail=f"Unsupported event_type: {payload.event_type}"
    )


@app.get("/compute-jobs/notifications")
async def get_compute_job_notifications(thread_id: str) -> dict:
    with _compute_jobs_lock:
        notifications = list(_thread_notifications.get(thread_id, []))
    return {"notifications": notifications}


@app.post("/upload")
async def upload_file(
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    replace_data: bool = Form(True),
):
    try:
        contents = await file.read()
        extension = os.path.splitext(file.filename)[1].lower()
        _save_uploaded_source_file(thread_id, file.filename, contents)
        dataframe = _load_dataframe_from_bytes(contents, extension)
        _save_uploaded_dataframe(thread_id, dataframe, replace_data)
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(dataframe),
            "columns": list(dataframe.columns),
            "shape": list(dataframe.shape),
            "replaced_data": replace_data,
            "message": f"Загружено {len(dataframe)} строк, {len(dataframe.columns)} колонок",
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {exc}") from exc


@app.get("/dataframes")
async def list_dataframes(thread_id: str = ""):
    dataframes = []
    if os.path.exists(current_path):
        current_dataframe = pd.read_pickle(current_path)
        dataframes.append(
            _build_dataframe_info("current", current_path, current_dataframe, True)
        )
    if thread_id:
        extra_path = os.path.join(
            DATAFRAMES_DIR, f"{EXTRA_DATAFRAME_PREFIX}{thread_id}.pkl"
        )
        if os.path.exists(extra_path):
            extra_dataframe = pd.read_pickle(extra_path)
            dataframes.append(
                _build_dataframe_info(
                    f"{EXTRA_DATAFRAME_PREFIX}{thread_id}",
                    extra_path,
                    extra_dataframe,
                    False,
                )
            )
    for file_name in sorted(os.listdir(DATAFRAMES_DIR)):
        if not file_name.startswith(COMPUTE_DATAFRAME_PREFIX) or not file_name.endswith(
            ".pkl"
        ):
            continue
        dataframe_path = os.path.join(DATAFRAMES_DIR, file_name)
        computed_dataframe = pd.read_pickle(dataframe_path)
        dataframes.append(
            _build_dataframe_info(
                file_name,
                dataframe_path,
                computed_dataframe,
                False,
            )
        )
    return {"dataframes": dataframes}


def run_server() -> None:
    print(f"Starting CSI Agent API on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    run_server()
