"""
Сервер CSI Agent на FastAPI с поддержкой SSE-стриминга.

Эндпоинты:
- POST /chat/stream — SSE-стриминг шагов агента в реальном времени
- POST /chat — классический JSON-ответ (обратная совместимость)
- POST /reset — сброс сессии и DataFrame
- GET /health — проверка работоспособности
- GET /sessions — список активных сессий

Запуск: python server.py
"""

import sys
import os
import io
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

import nest_asyncio
import uvicorn
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from load_data import get_data
from tools import DATAFRAMES_DIR, main_react_agent_sandbox
from langgraph_agent import CSIAgent, State

nest_asyncio.apply()

# ━━━━━━━━━━━━━━━━━━━ КОНФИГУРАЦИЯ ━━━━━━━━━━━━━━━━━━━
COMPUTE_TOOL_URL: str = ""
HOST: str = "127.0.0.1"
PORT: int = 8113


# ━━━━━━━━━━━━━━━━━━━ МОДЕЛИ ЗАПРОСА ━━━━━━━━━━━━━━━━━━━
class ChatRequest(BaseModel):
    """Входящий запрос от UI.

    Attributes:
        message: Текст сообщения пользователя.
        thread_id: Идентификатор сессии/диалога.
    """
    message: str
    thread_id: str


# ━━━━━━━━━━━━━━━━━━━ ИНИЦИАЛИЗАЦИЯ ━━━━━━━━━━━━━━━━━━━
print("📊 Загрузка данных...")
df = get_data()
os.makedirs(DATAFRAMES_DIR, exist_ok=True)

source_path = os.path.join(DATAFRAMES_DIR, "source_dataframe.pkl")
current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
df.to_pickle(source_path)
df.to_pickle(current_path)

print(f"✅ Данные загружены: shape={df.shape}")

_session_messages: Dict[str, List[str]] = {}
_session_flags: Dict[str, Dict[str, Any]] = {}


# ━━━━━━━━━━━━━━━━━━━ ХЕЛПЕРЫ ━━━━━━━━━━━━━━━━━━━
def clear_figures_from_sandbox(sandbox) -> None:
    """Удаляет все Plotly-фигуры из sandbox перед новым запросом.

    Args:
        sandbox: Экземпляр ClientPythonSandbox.
    """
    to_delete = [
        name
        for name, value in sandbox.globals.items()
        if isinstance(value, go.Figure)
    ]
    for name in to_delete:
        del sandbox.globals[name]


def extract_figures_from_sandbox(sandbox) -> List[Dict[str, str]]:
    """Извлекает уникальные Plotly-фигуры из sandbox.

    Args:
        sandbox: Экземпляр ClientPythonSandbox.

    Returns:
        Список словарей с name и figure_json.
    """
    figures: List[Dict[str, str]] = []
    seen_ids: set = set()

    for name, value in sandbox.globals.items():
        if name.startswith("_"):
            continue
        if isinstance(value, go.Figure):
            obj_id = id(value)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            try:
                figures.append({
                    "name": name,
                    "figure_json": pio.to_json(value),
                })
            except Exception as e:
                print(f"⚠️ Не удалось сериализовать фигуру {name}: {e}")

    return figures


def _get_or_create_session(thread_id: str) -> tuple:
    """Возвращает или создаёт сессию для thread_id.

    Args:
        thread_id: Идентификатор сессии.

    Returns:
        Кортеж (messages_list, flags_dict).
    """
    if thread_id not in _session_messages:
        _session_messages[thread_id] = []
        _session_flags[thread_id] = {
            "active_calculate_chain": False,
            "current_step_in_calculate_chain": 0,
            "last_activity": time.time(),
        }

    _session_flags[thread_id]["last_activity"] = time.time()
    return _session_messages[thread_id], _session_flags[thread_id]


def _cleanup_old_sessions(max_age_seconds: int = 3600) -> None:
    """Удаляет сессии старше max_age_seconds.

    Args:
        max_age_seconds: Максимальный возраст сессии в секундах.
    """
    now = time.time()
    expired = [
        tid
        for tid, flags in _session_flags.items()
        if now - flags.get("last_activity", 0) > max_age_seconds
    ]

    for tid in expired:
        _session_messages.pop(tid, None)
        _session_flags.pop(tid, None)

    if expired:
        print(f"🧹 Очищено {len(expired)} старых сессий")


def _make_safe_updates(updates: Optional[dict]) -> Dict[str, Any]:
    """Конвертирует обновления узла в JSON-безопасный формат.

    Args:
        updates: Словарь обновлений из узла графа.

    Returns:
        JSON-безопасный словарь.
    """
    if updates is None:
        return {}

    safe: Dict[str, Any] = {}
    for k, v in updates.items():
        if isinstance(v, pd.DataFrame):
            safe[k] = f"DataFrame shape={v.shape}"
        elif isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            safe[k] = str(type(v).__name__)

    return safe


def _get_current_df() -> pd.DataFrame:
    """Загружает текущий DataFrame из файла.

    Returns:
        Текущий DataFrame или исходный df.
    """
    if os.path.exists(current_path):
        return pd.read_pickle(current_path)
    return df


# ━━━━━━━━━━━━━━━━━━━ FASTAPI ━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="CSI Agent API",
    description="API для анализа данных с помощью AI-агента.",
    version="1.0.0",
)


# ─────────────── SSE-СТРИМИНГ ───────────────
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Стримит шаги агента как Server-Sent Events.

    Формат событий:
    - type=step — узел графа завершил работу
    - type=figure — создан Plotly-график
    - type=done — агент завершил, содержит final_answer
    - type=error — произошла ошибка

    Args:
        request: ChatRequest с message и thread_id.

    Returns:
        StreamingResponse с media_type text/event-stream.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            session_msgs, flags = _get_or_create_session(request.thread_id)
            session_msgs.append(" ##USER##: " + request.message)

            current_df = _get_current_df()
            clear_figures_from_sandbox(main_react_agent_sandbox)

            agent = CSIAgent(df, COMPUTE_TOOL_URL)
            final_answer: str = ""
            start_time = time.time()

            for chunk in agent.app.stream(
                    {
                        "user_input": " ".join(session_msgs[-7:]),
                        "active_calculate_chain": flags["active_calculate_chain"],
                        "current_step_in_calculate_chain": flags["current_step_in_calculate_chain"],
                        "start_time": start_time,
                        "last_change_df": current_df,
                    },
                    stream_mode="updates",
            ):
                for node_name, updates in chunk.items():
                    safe_updates = _make_safe_updates(updates)

                    if updates is not None:
                        if "final_answer" in updates:
                            final_answer = updates["final_answer"]
                        if "active_calculate_chain" in updates:
                            flags["active_calculate_chain"] = updates["active_calculate_chain"]
                        if "current_step_in_calculate_chain" in updates:
                            flags["current_step_in_calculate_chain"] = updates["current_step_in_calculate_chain"]

                    step_event = {
                        "type": "step",
                        "name": node_name,
                        "input": json.dumps(
                            safe_updates, ensure_ascii=False, default=str
                        )[:500],
                        "output": safe_updates.get(
                            "final_answer", safe_updates.get("react_agent_answer", "")
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }
                    yield f"data: {json.dumps(step_event, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

            figures = extract_figures_from_sandbox(main_react_agent_sandbox)
            for fig_data in figures:
                fig_event = {
                    "type": "figure",
                    "name": fig_data["name"],
                    "figure_json": fig_data["figure_json"],
                }
                yield f"data: {json.dumps(fig_event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            session_msgs.append(" ##AI AGENT##:" + final_answer)

            done_event = {
                "type": "done",
                "output": final_answer,
                "duration_seconds": round(time.time() - start_time, 2),
            }
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

            if len(_session_messages) > 100:
                _cleanup_old_sessions()

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_event = {
                "type": "error",
                "output": f"Ошибка: {str(e)}",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────── КЛАССИЧЕСКИЙ JSON-ОТВЕТ ───────────────
@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> dict:
    """Классический JSON-эндпоинт (обратная совместимость).

    Args:
        request: ChatRequest с message и thread_id.

    Returns:
        Словарь с output, steps и figures.
    """
    try:
        session_msgs, flags = _get_or_create_session(request.thread_id)
        session_msgs.append(" ##USER##: " + request.message)

        current_df = _get_current_df()
        clear_figures_from_sandbox(main_react_agent_sandbox)

        agent = CSIAgent(df, COMPUTE_TOOL_URL)
        steps: List[dict] = []
        final_answer: str = ""
        start_time = time.time()

        for chunk in agent.app.stream(
                {
                    "user_input": " ".join(session_msgs[-7:]),
                    "active_calculate_chain": flags["active_calculate_chain"],
                    "current_step_in_calculate_chain": flags["current_step_in_calculate_chain"],
                    "start_time": start_time,
                    "last_change_df": current_df,
                },
                stream_mode="updates",
        ):
            for node_name, updates in chunk.items():
                safe_updates = _make_safe_updates(updates)
                step_info = {
                    "node": node_name,
                    "name": node_name,
                    "timestamp": datetime.now().isoformat(),
                    "input": json.dumps(
                        safe_updates, ensure_ascii=False, default=str
                    )[:500],
                    "output": safe_updates.get(
                        "final_answer", safe_updates.get("react_agent_answer", "")
                    ),
                }

                if updates is not None:
                    step_info["updates"] = safe_updates
                    if "final_answer" in updates:
                        final_answer = updates["final_answer"]
                    if "active_calculate_chain" in updates:
                        flags["active_calculate_chain"] = updates["active_calculate_chain"]
                    if "current_step_in_calculate_chain" in updates:
                        flags["current_step_in_calculate_chain"] = updates["current_step_in_calculate_chain"]

                steps.append(step_info)

        session_msgs.append(" ##AI AGENT##:" + final_answer)

        figures = extract_figures_from_sandbox(main_react_agent_sandbox)

        steps.append({
            "node": "_meta",
            "name": "_meta",
            "input": "",
            "output": "",
            "duration_seconds": round(time.time() - start_time, 2),
        })

        if len(_session_messages) > 100:
            _cleanup_old_sessions()

        return {
            "output": final_answer,
            "steps": steps,
            "figures": figures,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"output": f"Ошибка: {str(e)}", "steps": [], "figures": []}


# ─────────────── СЕРВИСНЫЕ ЭНДПОИНТЫ ───────────────
@app.get("/health")
async def health_check() -> dict:
    """Проверка работоспособности сервера.

    Returns:
        Словарь со статусом и информацией о данных.
    """
    return {
        "status": "ok",
        "dataframe_shape": list(df.shape),
        "active_sessions": len(_session_messages),
        "dataframes_dir": DATAFRAMES_DIR,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reset")
async def reset_session(request: ChatRequest) -> dict:
    """Сбрасывает сессию и возвращает DataFrame к исходному.

    Args:
        request: ChatRequest с thread_id (message игнорируется).

    Returns:
        Словарь с подтверждением сброса.
    """
    thread_id = request.thread_id
    _session_messages.pop(thread_id, None)
    _session_flags.pop(thread_id, None)

    df.to_pickle(current_path)
    main_react_agent_sandbox.reset(keep_base=True)
    clear_figures_from_sandbox(main_react_agent_sandbox)

    return {
        "status": "reset",
        "thread_id": thread_id,
        "message": "Сессия сброшена, DataFrame восстановлен",
    }


@app.get("/sessions")
async def list_sessions() -> dict:
    """Возвращает список активных сессий.

    Returns:
        Словарь с информацией о каждой сессии.
    """
    sessions_info: Dict[str, Any] = {}
    for tid in _session_messages:
        flags = _session_flags.get(tid, {})
        sessions_info[tid] = {
            "message_count": len(_session_messages[tid]),
            "active_calculate_chain": flags.get("active_calculate_chain", False),
            "last_activity": flags.get("last_activity", 0),
        }

    return {"sessions": sessions_info}


from fastapi import File, UploadFile, Form


# ─────────────── ЗАГРУЗКА ФАЙЛОВ ───────────────
@app.post("/upload")
async def upload_file(
        thread_id: str = Form(...),
        file: UploadFile = File(...),
        replace_data: bool = Form(True)  # Если True - заменяет основной DataFrame
):
    """Загружает файл (CSV, Excel, JSON, Pickle) и обновляет DataFrame.

    Args:
        thread_id: Идентификатор сессии.
        file: Загружаемый файл.
        replace_data: Заменить ли основной DataFrame (True) или добавить как дополнительный.

    Returns:
        Информация о загруженном файле и DataFrame.
    """
    try:
        # Создаём директорию для загрузок
        uploads_dir = os.path.join(DATAFRAMES_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        # Читаем содержимое файла
        contents = await file.read()
        filename = f"{thread_id}_{file.filename}"
        file_path = os.path.join(uploads_dir, filename)

        # Сохраняем файл
        with open(file_path, "wb") as f:
            f.write(contents)

        # Определяем формат и загружаем в DataFrame
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext == ".csv":
            new_df = pd.read_csv(io.BytesIO(contents))
        elif file_ext in [".xlsx", ".xls"]:
            new_df = pd.read_excel(io.BytesIO(contents))
        elif file_ext == ".json":
            new_df = pd.read_json(io.BytesIO(contents))
        elif file_ext in [".pkl", ".pickle"]:
            new_df = pd.read_pickle(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат: {file_ext}. Поддерживаются: CSV, Excel, JSON, Pickle"
            )

        # Обновляем DataFrame
        if replace_data:
            # Заменяем основной DataFrame
            new_df.to_pickle(current_path)
            print(f"✅ DataFrame заменён: shape={new_df.shape}")
        else:
            # Добавляем как дополнительный DataFrame
            extra_df_path = os.path.join(DATAFRAMES_DIR, f"extra_{thread_id}.pkl")
            new_df.to_pickle(extra_df_path)
            print(f"✅ Дополнительный DataFrame сохранён: shape={new_df.shape}")

        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(new_df),
            "columns": list(new_df.columns),
            "shape": list(new_df.shape),
            "replaced_data": replace_data,
            "message": f"Загружено {len(new_df)} строк, {len(new_df.columns)} колонок"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")


@app.get("/dataframes")
async def list_dataframes(thread_id: str = ""):
    """Возвращает информацию о доступных DataFrame.

    Args:
        thread_id: Опциональный фильтр по сессии.

    Returns:
        Список DataFrame с метаданными.
    """
    dataframes = []

    # Основной DataFrame
    if os.path.exists(current_path):
        df = pd.read_pickle(current_path)
        dataframes.append({
            "name": "current",
            "path": current_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "is_current": True
        })

    # Дополнительные DataFrame
    if thread_id:
        extra_path = os.path.join(DATAFRAMES_DIR, f"extra_{thread_id}.pkl")
        if os.path.exists(extra_path):
            df = pd.read_pickle(extra_path)
            dataframes.append({
                "name": f"extra_{thread_id}",
                "path": extra_path,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "is_current": False
            })

    return {"dataframes": dataframes}


# ━━━━━━━━━━━━━━━━━━━ ЗАПУСК ━━━━━━━━━━━━━━━━━━━
def run_server() -> None:
    """Запускает uvicorn-сервер."""
    print(f"🚀 Запуск CSI Agent API на {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()