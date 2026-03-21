import sys
import os
import io

# Очистка конфликтных модулей
sys.modules.pop('typing_extensions', None)
sys.modules.pop('opentelemetry', None)

import json
import asyncio
import plotly.io as pio
import plotly.graph_objects as go
import chainlit as cl
import httpx
import uuid

# ─── КОНФИГУРАЦИЯ ───
THREAD_ID = str(uuid.uuid4())
API_URL = "http://127.0.0.1:8113/chat/stream"
BASE_API_URL = "http://127.0.0.1:8113"

from chainlit.input_widget import Switch

node_name_map = {
    "add_inform_in_user_input": "Анализирую запрос пользователя...",
    "cheking_for_common_request": "Проверяю, нужны ли уточняющие запросы...",
    "cheking_that_query_belongs_table": "Анализирую структуру таблицы...",
    "cheking_need_make_additional_question": "Проверяю, хватает ли мне данных...",
    "re_act_agent": "Генерирую код и выполняю преобразования...",
    "re_act_stat_agent": "Вычисляю статистику..."
}


# ─── ФУНКЦИЯ ЗАГРУЗКИ ФАЙЛА ───
async def handle_file_upload():
    """Запрашивает файл у пользователя и загружает на сервер"""
    files = await cl.AskFileMessage(
        content="📁 Выберите файл для загрузки (CSV, Excel, JSON, Pickle)",
        accept={
            "text/csv": [".csv"],
            "application/vnd.ms-excel": [".xls"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
            "application/json": [".json"],
            "application/octet-stream": [".pkl", ".pickle"],
        },
        max_size_mb=50,
    ).send()

    if not files:
        return

    file = files[0]
    msg = cl.Message(content=f"📤 Загрузка {file.name}...")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(file.path, "rb") as f:
                response = await client.post(
                    f"{BASE_API_URL}/upload",
                    files={"file": (file.name, f, file.type)},
                    data={
                        "thread_id": cl.user_session.get("id"),
                        "replace_data": "true"
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    await msg.stream_token(f"\n✅ {file.name} загружен")
                    await msg.stream_token(f"\n📊 {result['rows']} строк × {result['shape'][1]} колонок")
                else:
                    await msg.stream_token(f"\n❌ Ошибка: {response.text}")
    except Exception as e:
        await msg.stream_token(f"\n❌ Ошибка соединения: {str(e)}")


# ─── СТАРТ ЧАТА ───
@cl.on_chat_start
async def start():
    await cl.ChatSettings([
        Switch(id="show_tools", label="Показывать работу инструментов", initial=True)
    ]).send()

    await cl.Message(
        content="""
👋 **Привет! Я CSI Agent для анализа данных.**

💡 **Возможности:**
- 📎 Напишите **/upload** для загрузки файла
- 📊 Получайте графики и таблицы в ответ
- ⚙️ Настройте отображение шагов в меню справа

**Поддерживаемые форматы:** CSV, Excel, JSON, Pickle
**Максимальный размер:** 50MB
"""
    ).send()

    cl.user_session.set("id", THREAD_ID)


# ─── ОСНОВНОЙ ОБРАБОТЧИК СООБЩЕНИЙ ───
@cl.on_message
async def main(message: cl.Message):
    # ─── ОБРАБОТКА КОМАНД ───
    if message.content.strip() == "/upload":
        await handle_file_upload()
        return

    settings = cl.user_session.get("chat_settings")
    show_tools = settings.get("show_tools", True) if settings else True

    active_steps = {}
    step_counter = 0

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
                "POST",
                API_URL,
                json={
                    "message": message.content,
                    "thread_id": cl.user_session.get("id"),
                },
        ) as response:
            async for line in response.aiter_lines():
                # Проверка на SSE формат
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: "):]

                # Пропускаем пустые строки
                if not raw.strip():
                    continue

                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                # ─── ШАГ АГЕНТА ───
                if event_type == "step":
                    node_name = event.get("name", "unknown")
                    node_name = node_name_map.get(node_name, node_name)

                    if show_tools:
                        step_counter += 1
                        step = cl.Step(
                            name=f"🔵 {node_name}",
                            type="run",
                        )
                        await step.send()
                        active_steps[step_counter] = step

                # ─── ТОКЕНЫ (ПОТОКОВАЯ ПЕЧАТЬ) ───
                elif event_type == "token":
                    if show_tools and active_steps:
                        # Отправляем токен в последний активный шаг
                        last_step = active_steps[step_counter]
                        await last_step.stream_token(event.get("content", ""))

                # ─── ГРАФИК ───
                elif event_type == "figure":
                    fig_name = event.get("name", "chart")
                    fig_json = event.get("figure_json", "")

                    if fig_json:
                        fig = pio.from_json(fig_json)
                        await cl.Message(
                            content=f"📊 График: {fig_name}",
                            elements=[
                                cl.Plotly(
                                    name=fig_name,
                                    figure=fig,
                                    display="inline",
                                )
                            ],
                        ).send()

                # ─── ФИНАЛЬНЫЙ ОТВЕТ ───
                elif event_type == "done":
                    # Удаляем шаги, если они были показаны
                    if show_tools and active_steps:
                        await asyncio.gather(
                            *[step.remove() for step in active_steps.values()],
                            return_exceptions=True
                        )
                        await asyncio.sleep(0.05)

                    output = event.get("output", "Нет ответа")
                    duration = event.get("duration_seconds", 0)

                    await cl.Message(
                        content=f"{output}\n\n⏱ {duration}с"
                    ).send()

                # ─── ОШИБКА ───
                elif event_type == "error":
                    # Удаляем шаги при ошибке
                    if show_tools and active_steps:
                        await asyncio.gather(
                            *[step.remove() for step in active_steps.values()],
                            return_exceptions=True
                        )

                    await cl.Message(
                        content=f"❌ {event.get('output', 'Неизвестная ошибка')}"
                    ).send()