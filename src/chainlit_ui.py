import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any
import chainlit as cl
import httpx
import plotly.io as pio
from chainlit.context import ChainlitContext, get_context
from chainlit.input_widget import Switch
from chainlit.utils import utc_now

API_URL = "http://127.0.0.1:8113/chat/stream"
BASE_API_URL = "http://127.0.0.1:8113"
UPLOAD_TIMEOUT_SECONDS = 300
CHAT_TIMEOUT_SECONDS = 600
STEP_REMOVE_DELAY_SECONDS = 0.05
NOTIFICATION_POLL_INTERVAL_SECONDS = 1
SHOW_TOOLS_SETTING_ID = "show_tools"
BACKEND_THREAD_ID_KEY = "backend_thread_id"
NOTIFICATION_TASK_KEY = "notification_task"
PENDING_COMPUTE_JOBS_KEY = "pending_compute_jobs"
COMPUTE_JOB_ID_PATTERN = re.compile(
    r"Идентификатор задачи:\s*(?P<job_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
DEFAULT_AGENT_OUTPUT = "Нет ответа"
DEFAULT_AGENT_ERROR = "Неизвестная ошибка"
COMPUTE_NOTIFICATION_AUTHOR = "Compute Tool"
WELCOME_MESSAGE = """
**Привет! Это CSI Agent для анализа данных.**
Возможности:
- прикрепляйте файл прямо к сообщению через встроенное вложение Chainlit;
- получайте текстовые ответы и графики;
- включайте или скрывайте шаги работы агента через настройки справа.
Поддерживаемые форматы: CSV, Excel, JSON, Pickle
Максимальный размер: 50MB
""".strip()
NODE_NAME_MAP = {
    "request_started": "Запрос принят, запускаю обработку...",
    "start_normalization_chain": "Строю draft нормализации...",
    "continue_normalization_chain": "Обновляю или применяю draft нормализации...",
    "add_inform_in_user_input": "Анализирую запрос пользователя...",
    "cheking_for_common_request": "Проверяю, нужны ли уточнения...",
    "cheking_that_query_belongs_table": "Сопоставляю запрос со структурой таблицы...",
    "cheking_need_make_additional_question": "Проверяю, достаточно ли данных для ответа...",
    "re_act_agent": "Генерирую код и запускаю преобразования...",
    "re_act_stat_agent": "Считаю статистику...",
}
NODE_NAME_MAP["checking_for_common_request"] = NODE_NAME_MAP[
    "cheking_for_common_request"
]
_seen_notification_ids: dict[str, set[str]] = {}


def _get_thread_id() -> str:
    return cl.user_session.get(BACKEND_THREAD_ID_KEY, "")


def _get_show_tools() -> bool:
    settings = cl.user_session.get("chat_settings") or {}
    return settings.get(SHOW_TOOLS_SETTING_ID, True)


def _get_pending_compute_job_ids() -> set[str]:
    pending_jobs = cl.user_session.get(PENDING_COMPUTE_JOBS_KEY)
    if isinstance(pending_jobs, set):
        return pending_jobs
    normalized_pending_jobs = {
        str(job_id).strip() for job_id in (pending_jobs or []) if str(job_id).strip()
    }
    cl.user_session.set(PENDING_COMPUTE_JOBS_KEY, normalized_pending_jobs)
    return normalized_pending_jobs


def _track_pending_compute_job(message: str) -> str | None:
    match = COMPUTE_JOB_ID_PATTERN.search(message or "")
    if match is None:
        return None
    job_id = match.group("job_id")
    _get_pending_compute_job_ids().add(job_id)
    return job_id


def _stop_notification_task() -> None:
    task = cl.user_session.get(NOTIFICATION_TASK_KEY)
    if task is not None and not task.done():
        task.cancel()
    cl.user_session.set(NOTIFICATION_TASK_KEY, None)


def _ensure_notification_task(saved_context: ChainlitContext) -> None:
    thread_id = _get_thread_id().strip()
    if not thread_id or not _get_pending_compute_job_ids():
        return
    task = cl.user_session.get(NOTIFICATION_TASK_KEY)
    if task is not None and not task.done():
        return
    task = asyncio.create_task(_poll_compute_notifications(thread_id, saved_context))
    cl.user_session.set(NOTIFICATION_TASK_KEY, task)


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    payload = line[len("data: ") :].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_attached_files(message: cl.Message) -> list[Any]:
    return [
        element
        for element in (message.elements or [])
        if getattr(element, "path", None)
    ]


async def _remove_active_steps(active_steps: dict[int, cl.Step]) -> None:
    if not active_steps:
        return
    await asyncio.gather(
        *[step.remove() for step in active_steps.values()],
        return_exceptions=True,
    )
    active_steps.clear()


async def _stream_message(message: cl.Message, *tokens: str) -> None:
    for token in tokens:
        await message.stream_token(token)


async def _upload_file_element(file_element: Any) -> None:
    file_path = Path(file_element.path)
    file_name = getattr(file_element, "name", file_path.name)
    file_type = getattr(file_element, "mime", None) or "application/octet-stream"
    status_message = cl.Message(content=f"Загрузка {file_name}...")
    await status_message.send()
    try:
        async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
            with open(file_path, "rb") as file_obj:
                response = await client.post(
                    f"{BASE_API_URL}/upload",
                    files={"file": (file_name, file_obj, file_type)},
                    data={"thread_id": _get_thread_id(), "replace_data": "true"},
                )
        if response.status_code == 200:
            result = response.json()
            await _stream_message(
                status_message,
                f"\nФайл {file_name} загружен",
                f"\n{result['rows']} строк x {result['shape'][1]} колонок",
            )
            return
        await _stream_message(status_message, f"\nОшибка: {response.text}")
    except Exception as exc:
        await _stream_message(status_message, f"\nОшибка соединения: {exc}")


async def _handle_attached_files(message: cl.Message) -> None:
    attached_files = _extract_attached_files(message)
    if not attached_files:
        return
    if len(attached_files) > 1:
        await cl.Message(
            content="Сейчас загружается только первый прикрепленный файл. Остальные вложения проигнорированы."
        ).send()
    await _upload_file_element(attached_files[0])


async def _handle_step_event(
    event: dict[str, Any],
    active_steps: dict[int, cl.Step],
    step_counter: int,
    show_tools: bool,
) -> int:
    if not show_tools:
        return step_counter
    step_counter += 1
    node_name = event.get("name", "unknown")
    step_title = NODE_NAME_MAP.get(node_name, node_name)
    step = cl.Step(name=step_title, type="run")
    await step.send()
    active_steps[step_counter] = step
    return step_counter


async def _handle_token_event(
    event: dict[str, Any],
    active_steps: dict[int, cl.Step],
    step_counter: int,
    show_tools: bool,
) -> None:
    if not show_tools or not active_steps or step_counter not in active_steps:
        return
    await active_steps[step_counter].stream_token(event.get("content", ""))


async def _handle_figure_event(event: dict[str, Any]) -> None:
    figure_json = event.get("figure_json")
    if not figure_json:
        return
    figure_name = event.get("name", "chart")
    figure = pio.from_json(figure_json)
    await cl.Message(
        content=f"График: {figure_name}",
        elements=[
            cl.Plotly(
                name=figure_name,
                figure=figure,
                display="inline",
            )
        ],
    ).send()


async def _handle_done_event(
    event: dict[str, Any],
    active_steps: dict[int, cl.Step],
    show_tools: bool,
    saved_context: ChainlitContext,
) -> None:
    if show_tools:
        await _remove_active_steps(active_steps)
        await asyncio.sleep(STEP_REMOVE_DELAY_SECONDS)
    output = event.get("output", DEFAULT_AGENT_OUTPUT)
    duration_seconds = event.get("duration_seconds", 0)
    await cl.Message(content=f"{output}\n\n{duration_seconds}с").send()
    if _track_pending_compute_job(output):
        _ensure_notification_task(saved_context)


async def _handle_error_event(
    event: dict[str, Any],
    active_steps: dict[int, cl.Step],
    show_tools: bool,
) -> None:
    if show_tools:
        await _remove_active_steps(active_steps)
    await cl.Message(
        content=f"Ошибка: {event.get('output', DEFAULT_AGENT_ERROR)}"
    ).send()


def _get_seen_notification_ids(thread_id: str) -> set[str]:
    return _seen_notification_ids.setdefault(thread_id, set())


async def _send_compute_success_message(
    saved_context: ChainlitContext,
    content: str,
) -> None:
    created_at = utc_now()
    step_dict = {
        "id": str(uuid.uuid4()),
        "threadId": saved_context.session.thread_id,
        "parentId": None,
        "createdAt": created_at,
        "command": None,
        "modes": None,
        "start": created_at,
        "end": created_at,
        "output": content,
        "name": COMPUTE_NOTIFICATION_AUTHOR,
        "type": "assistant_message",
        "language": None,
        "streaming": False,
        "isError": False,
        "waitForAnswer": False,
        "metadata": {},
        "tags": None,
    }
    await saved_context.emitter.send_step(step_dict)


async def _emit_compute_notification(
    thread_id: str,
    notification: dict[str, Any],
    saved_context: ChainlitContext,
) -> None:
    del thread_id
    status = notification.get("status")
    message = notification.get("message", "")
    await saved_context.emitter.send_toast(
        message or "Завершилось фоновое вычисление.",
        "success"
        if status == "success"
        else ("error" if status == "error" else "info"),
    )
    if status == "success" and message:
        await _send_compute_success_message(saved_context, message)


async def _poll_compute_notifications(
    thread_id: str,
    saved_context: ChainlitContext,
) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        seen_notification_ids = _get_seen_notification_ids(thread_id)
        pending_job_ids = _get_pending_compute_job_ids()
        while True:
            try:
                response = await client.get(
                    f"{BASE_API_URL}/compute-jobs/notifications",
                    params={"thread_id": thread_id},
                )
                response.raise_for_status()
                payload = response.json()
                for notification in payload.get("notifications", []):
                    notification_id = notification.get("id")
                    if notification_id and notification_id in seen_notification_ids:
                        continue
                    if notification_id:
                        seen_notification_ids.add(notification_id)
                    await _emit_compute_notification(
                        thread_id, notification, saved_context
                    )
                    job_id = str(notification.get("job_id", "")).strip()
                    if job_id and notification.get("status") in {"success", "error"}:
                        pending_job_ids.discard(job_id)
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                pass
            except Exception as exc:
                print(
                    f"Chainlit notification polling error: {type(exc).__name__}: {exc!r}",
                    flush=True,
                )
            if not pending_job_ids:
                break
            await asyncio.sleep(NOTIFICATION_POLL_INTERVAL_SECONDS)


@cl.on_chat_start
async def start() -> None:
    thread_id = str(uuid.uuid4())
    cl.user_session.set(BACKEND_THREAD_ID_KEY, thread_id)
    cl.user_session.set(PENDING_COMPUTE_JOBS_KEY, set())
    _stop_notification_task()
    await cl.ChatSettings(
        [
            Switch(
                id=SHOW_TOOLS_SETTING_ID,
                label="Показывать работу инструментов",
                initial=True,
            )
        ]
    ).send()
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    _stop_notification_task()
    cl.user_session.set(PENDING_COMPUTE_JOBS_KEY, set())
    thread_id = cl.user_session.get(BACKEND_THREAD_ID_KEY, "")
    if thread_id:
        _seen_notification_ids.pop(thread_id, None)


@cl.on_message
async def main(message: cl.Message) -> None:
    await _handle_attached_files(message)
    if not (message.content or "").strip():
        return
    saved_context = get_context()
    show_tools = _get_show_tools()
    active_steps: dict[int, cl.Step] = {}
    step_counter = 0
    try:
        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST",
                API_URL,
                json={"message": message.content, "thread_id": _get_thread_id()},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None:
                        continue
                    event_type = event.get("type")
                    if event_type == "step":
                        step_counter = await _handle_step_event(
                            event,
                            active_steps,
                            step_counter,
                            show_tools,
                        )
                    elif event_type == "token":
                        await _handle_token_event(
                            event, active_steps, step_counter, show_tools
                        )
                    elif event_type == "figure":
                        await _handle_figure_event(event)
                    elif event_type == "done":
                        await _handle_done_event(
                            event, active_steps, show_tools, saved_context
                        )
                    elif event_type == "error":
                        await _handle_error_event(event, active_steps, show_tools)
    except Exception as exc:
        if show_tools:
            await _remove_active_steps(active_steps)
        await cl.Message(content=f"Ошибка соединения с backend: {exc}").send()
