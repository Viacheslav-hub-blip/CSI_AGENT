import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import aiofiles
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from compute_tool import ProcessDataFrameAgent
from model import GigaChat_Max

logger = logging.getLogger(__name__)
BASE_STORAGE_DIR = Path(__file__).resolve().parent.parent / "compute_tool_storage"
UPLOAD_DIR = BASE_STORAGE_DIR / "uploads"
RESULT_DIR = BASE_STORAGE_DIR / "results"
MAX_DATAFRAME_ROWS = 500
UPLOAD_CHUNK_SIZE = 1024 * 64
STREAM_EVENT_DELAY_SECONDS = 0.15
HEARTBEAT_INTERVAL_SECONDS = 15
QUEUE_END_MARKER = "__end__"
JOB_RETENTION_SECONDS = 3600
CALLBACK_REQUEST_TIMEOUT_SECONDS = 15
PROGRESS_EVENT_THROTTLE_SECONDS = 0.75
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Compute Tool API", version="1.0.0")
JOB_QUEUES: dict[str, asyncio.Queue[str]] = {}
JOB_STATUS: dict[str, str] = {}
JOB_CREATED_AT: dict[str, datetime] = {}
JOB_TASKS: dict[str, asyncio.Task] = {}


def _emit_console_status(job_id: str, message: str) -> None:
    normalized_message = message.strip()
    if not normalized_message:
        return
    logger.info("[compute_job=%s] %s", job_id, normalized_message)
    print(f"[compute_job={job_id}] {normalized_message}", flush=True)


async def sse_event(data: str, event: str | None = None) -> str:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for chunk in data.splitlines():
        lines.append(f"data: {chunk}")
    lines.append("")
    logger.info("##SERVER_SSE_EVENT##", extra={"extra_info": " ".join(lines)})
    return "\n".join(lines) + "\n"


async def _queue_event(
    queue: asyncio.Queue[str],
    payload: dict[str, Any],
    event_name: str,
) -> None:
    await queue.put(await sse_event(json.dumps(payload), event=event_name))
    await asyncio.sleep(STREAM_EVENT_DELAY_SECONDS)


def _cleanup_old_jobs() -> None:
    now = datetime.now()
    expired_job_ids = [
        job_id
        for job_id, created_at in JOB_CREATED_AT.items()
        if (now - created_at).total_seconds() > JOB_RETENTION_SECONDS
    ]
    for job_id in expired_job_ids:
        JOB_CREATED_AT.pop(job_id, None)
        JOB_STATUS.pop(job_id, None)
        JOB_QUEUES.pop(job_id, None)
        JOB_TASKS.pop(job_id, None)


def _get_result_path(job_id: str) -> Path:
    return RESULT_DIR / f"{job_id}.pkl"


def _get_result_url(job_id: str) -> str:
    return f"/results/{job_id}/result.pkl"


async def _persist_upload(upload_file: UploadFile, save_path: Path) -> None:
    async with aiofiles.open(save_path, "wb") as output_file:
        while True:
            chunk = await upload_file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            await output_file.write(chunk)


async def process_dataframe(
    dataframe: pd.DataFrame,
    user_input: str,
    queue: asyncio.Queue[str],
    job_id: str,
    callback_url: str,
    callback_thread_id: str,
) -> pd.DataFrame:
    logger.info("##SERVER_PROCESS_DATAFRAME##")
    status_reporter = _build_status_reporter(
        queue=queue,
        job_id=job_id,
        callback_url=callback_url,
        callback_thread_id=callback_thread_id,
    )
    progress_reporter = _build_progress_reporter(
        queue=queue,
        job_id=job_id,
        callback_url=callback_url,
        callback_thread_id=callback_thread_id,
        total_rows=len(dataframe),
    )
    agent = ProcessDataFrameAgent(
        dataframe,
        GigaChat_Max,
        progress_reporter=progress_reporter,
        status_reporter=status_reporter,
    )
    async for chunk in agent.run_with_streaming({"user_input": user_input}):
        status_reporter(chunk)
        await _queue_event(queue, {"type": "info", "msg": chunk}, "info")
    return agent.current_dataframe


async def _notify_callback(callback_url: str, payload: dict[str, Any]) -> None:
    if not callback_url:
        return

    def _send_request() -> None:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=CALLBACK_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    try:
        await asyncio.to_thread(_send_request)
    except Exception as exc:
        logger.warning("Failed to send compute callback: %s", exc)


async def _emit_progress_update(
    queue: asyncio.Queue[str],
    job_id: str,
    callback_url: str,
    callback_thread_id: str,
    processed_rows: int | None,
    total_rows: int | None,
    message: str,
) -> None:
    payload: dict[str, Any] = {
        "type": "progress",
        "job_id": job_id,
        "message": message,
    }
    if processed_rows is not None:
        payload["processed_rows"] = processed_rows
    if total_rows is not None:
        payload["total_rows"] = total_rows
    await _queue_event(queue, payload, "progress")
    await _notify_callback(
        callback_url,
        {
            "event_type": "progress",
            "thread_id": callback_thread_id,
            "job_id": job_id,
            "processed_rows": processed_rows,
            "total_rows": total_rows,
            "message": message,
        },
    )


def _build_progress_reporter(
    queue: asyncio.Queue[str],
    job_id: str,
    callback_url: str,
    callback_thread_id: str,
    total_rows: int,
) -> Callable[[int, int, str], None]:
    loop = asyncio.get_running_loop()
    progress_state = {
        "last_sent_at": 0.0,
        "last_processed": -1,
        "last_message": "",
    }

    def report_progress(processed_rows: int, total_rows_arg: int, message: str) -> None:
        try:
            processed = max(int(processed_rows), 0)
        except (TypeError, ValueError):
            processed = 0
        try:
            total = int(total_rows_arg)
        except (TypeError, ValueError):
            total = total_rows
        total = max(total, 0)
        normalized_message = str(message or "").strip() or "Идет обработка строк"
        current_time = time.monotonic()
        is_initial = processed == 0
        is_final = total == 0 or processed >= total
        message_changed = normalized_message != progress_state["last_message"]
        processed_changed = processed != progress_state["last_processed"]
        should_send = (
            is_initial
            or is_final
            or message_changed
            or (
                processed_changed
                and current_time - progress_state["last_sent_at"]
                >= PROGRESS_EVENT_THROTTLE_SECONDS
            )
        )
        if not should_send:
            return
        progress_state["last_sent_at"] = current_time
        progress_state["last_processed"] = processed
        progress_state["last_message"] = normalized_message
        future = asyncio.run_coroutine_threadsafe(
            _emit_progress_update(
                queue=queue,
                job_id=job_id,
                callback_url=callback_url,
                callback_thread_id=callback_thread_id,
                processed_rows=processed,
                total_rows=total,
                message=normalized_message,
            ),
            loop,
        )
        future.add_done_callback(
            lambda done: done.exception() if done.exception() else None
        )

    return report_progress


def _build_status_reporter(
    queue: asyncio.Queue[str],
    job_id: str,
    callback_url: str,
    callback_thread_id: str,
) -> Callable[[str], None]:
    loop = asyncio.get_running_loop()
    last_message = ""

    def report_status(message: str) -> None:
        nonlocal last_message
        normalized_message = str(message or "").strip()
        if not normalized_message or normalized_message == last_message:
            return
        last_message = normalized_message
        _emit_console_status(job_id, normalized_message)
        future = asyncio.run_coroutine_threadsafe(
            _emit_progress_update(
                queue=queue,
                job_id=job_id,
                callback_url=callback_url,
                callback_thread_id=callback_thread_id,
                processed_rows=None,
                total_rows=None,
                message=normalized_message,
            ),
            loop,
        )
        future.add_done_callback(
            lambda done: done.exception() if done.exception() else None
        )

    return report_status


async def worker_compute(
    job_id: str,
    dataframe_path: Path,
    user_input: str,
    callback_url: str,
    callback_thread_id: str,
) -> None:
    logger.info(
        "##WORKER_COMPUTE##",
        extra={"extra_info": f"job_id={job_id}; user_input={user_input}"},
    )
    queue = JOB_QUEUES[job_id]
    loop = asyncio.get_running_loop()
    try:
        JOB_STATUS[job_id] = "running"
        _emit_console_status(job_id, "Задача принята в обработку")
        await _queue_event(
            queue, {"type": "info", "msg": "Loading input pickle"}, "info"
        )
        _emit_console_status(job_id, "Загружаю входной DataFrame")
        dataframe = await loop.run_in_executor(
            None, pd.read_pickle, str(dataframe_path)
        )
        if len(dataframe) > MAX_DATAFRAME_ROWS:
            raise ValueError("DataFrame is too large for compute_tool service")
        await _emit_progress_update(
            queue=queue,
            job_id=job_id,
            callback_url=callback_url,
            callback_thread_id=callback_thread_id,
            processed_rows=0,
            total_rows=len(dataframe),
            message="Задача принята, начинаю обработку DataFrame",
        )
        _emit_console_status(job_id, "Запускаю агент вычислений")
        result_dataframe = await process_dataframe(
            dataframe,
            user_input,
            queue,
            job_id,
            callback_url,
            callback_thread_id,
        )
        result_path = _get_result_path(job_id)
        _emit_console_status(job_id, "Сохраняю вычисленный DataFrame")
        await _queue_event(queue, {"type": "info", "msg": "Saving result"}, "info")
        await loop.run_in_executor(None, result_dataframe.to_pickle, str(result_path))
        await _queue_event(
            queue,
            {"type": "result", "job_id": job_id, "result_url": _get_result_url(job_id)},
            "result",
        )
        await _notify_callback(
            callback_url,
            {
                "event_type": "result",
                "thread_id": callback_thread_id,
                "job_id": job_id,
                "result_url": _get_result_url(job_id),
                "row_count": len(result_dataframe),
                "column_count": len(result_dataframe.columns),
            },
        )
        JOB_STATUS[job_id] = "done"
        _emit_console_status(job_id, "Вычисление успешно завершено")
    except Exception as exc:
        JOB_STATUS[job_id] = "error"
        _emit_console_status(job_id, f"Вычисление завершилось ошибкой: {exc}")
        await _queue_event(queue, {"type": "error", "error": str(exc)}, "error")
        await _notify_callback(
            callback_url,
            {
                "event_type": "error",
                "thread_id": callback_thread_id,
                "job_id": job_id,
                "error": str(exc),
            },
        )
    finally:
        await queue.put(QUEUE_END_MARKER)
        await asyncio.sleep(STREAM_EVENT_DELAY_SECONDS)
        JOB_TASKS.pop(job_id, None)
        try:
            if dataframe_path.exists():
                dataframe_path.unlink()
        except OSError:
            logger.warning("Failed to remove temp upload %s", dataframe_path)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    user_input: str = Form(...),
    callback_url: str = Form(""),
    callback_thread_id: str = Form(""),
):
    logger.info("##CREATE_JOB##", extra={"extra_info": user_input})
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_input.pkl"
    _emit_console_status(job_id, "Получен новый job на вычисление")
    try:
        await _persist_upload(file, save_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save upload file: {exc}",
        ) from exc
    JOB_QUEUES[job_id] = asyncio.Queue()
    JOB_STATUS[job_id] = "queued"
    JOB_CREATED_AT[job_id] = datetime.now()
    JOB_TASKS[job_id] = asyncio.create_task(
        worker_compute(
            job_id,
            save_path,
            user_input,
            callback_url.strip(),
            callback_thread_id.strip(),
        )
    )
    return {
        "job_id": job_id,
        "sse_url": f"/events/{job_id}",
        "result_hint": _get_result_url(job_id),
    }


@app.get("/events/{job_id}")
async def events(job_id: str):
    logger.info("##EVENTS##", extra={"extra_info": f"JOB_ID: {job_id}"})
    _cleanup_old_jobs()
    queue = JOB_QUEUES.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_generator():
        yield await sse_event(
            json.dumps({"type": "connected", "job_id": job_id}),
            event="connected",
        )
        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if payload == QUEUE_END_MARKER:
                yield await sse_event(json.dumps({"type": "closed"}), event="closed")
                break
            yield payload

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    _cleanup_old_jobs()
    status = JOB_STATUS.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    payload = {"job_id": job_id, "status": status}
    if status == "done":
        payload["result_url"] = _get_result_url(job_id)
    return payload


@app.get("/results/{job_id}/result.pkl")
async def download_result(job_id: str):
    result_path = _get_result_path(job_id)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(
        result_path,
        media_type="application/octet-stream",
        filename=f"{job_id}_result.pkl",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8200)
