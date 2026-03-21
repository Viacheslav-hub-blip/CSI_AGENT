"""
Главный модуль агента: граф состояний (LangGraph), маршрутизация, вызовы суб-агентов и интерактивный цикл.
CSIAgent — оркестратор, который:
1. Классифицирует запрос пользователя.
2. Маршрутизирует по узлам графа.
3. Вызывает суб-агентов с нужными инструментами.
4. Формирует итоговый ответ.
"""

import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
import re
import uuid
import time
import json
import requests
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

from model import GigaChat_Max, GigaChat_Pro
from prompts import (
    prompt_react,
    prompt_react_stats,
    promt_add_inform_in_query,
    prompt_define_type_user_input,
    prompt_cheking_that_query_belongs_table,
    prompt_need_make_additional_question,
    prompt_make_additional_question,
    prompt_varios_purpose_agent,
)
from sub_agents import (
    get_react_main_agent,
    get_react_stat_agent,
    get_react_purpose_agent,
)
from tools import (
    main_react_agent_sandbox,
    load_dataframe_to_sandbox,
    get_dataframe_from_sandbox,
    DATAFRAMES_DIR,
)

# ━━━━━━━━━━━━━━━━━━━ КОНСТАНТЫ ━━━━━━━━━━━━━━━━━━━
CONNECTION_CLOSED_BY_SERVER: str = "Соединение закрыто сервером"
EVENT: str = "Событие:"
HAS_RESULT: str = "Получен результат"
RESULT_URL: str = "URL результата:"
SAVED_RESULT: str = "Результат сохранен"
TIMEOUT: str = "Таймаут"
INVOKE_AGENT_LLM_ERROR: str = "Ошибка при вызове агента"
QUERY_DOESNT_RELATE_TABLE: str = "Запрос не относится к таблице"
FINAL_ANSWER_PROMPT: str = "Сформируйте итоговый ответ на основе контекста"
MAX_CONTEXT_LENGTH: int = 130_000


# ━━━━━━━━━━━━━━━━━━━ STATE ━━━━━━━━━━━━━━━━━━━
class State(TypedDict):
    """Состояние графа агента.

    Attributes:
        user_input: История сообщений (##USER##: / ##AI AGENT##:).
        last_change_df: Текущий DataFrame после обработки.
        react_agent_answer: Ответ Main Agent (код + результат).
        re_act_stat_agent_answer: Ответ Stat Agent (статистика).
        final_answer: Итоговый ответ пользователю.
        active_calculate_chain: Флаг цепочки внешних вычислений.
        current_step_in_calculate_chain: Шаг в цепочке (0/1/2).
        start_time: Время начала обработки.
    """
    user_input: str
    last_change_df: pd.DataFrame
    react_agent_answer: str
    re_act_stat_agent_answer: str
    final_answer: str
    active_calculate_chain: bool
    current_step_in_calculate_chain: int
    start_time: float


# ━━━━━━━━━━━━━━━━━━━ АГЕНТ ━━━━━━━━━━━━━━━━━━━
class CSIAgent:
    """Оркестратор-агент для анализа данных.

    Управляет графом состояний, маршрутизирует запросы, вызывает суб-агентов и работает с файлами DataFrame.

    Args:
        df: Исходный DataFrame.
        compute_tool_url: URL внешнего сервиса вычислений.
    """

    def __init__(self, df: pd.DataFrame, compute_tool_url: str) -> None:
        self.df: pd.DataFrame = df
        self.compute_tool_url: str = compute_tool_url
        self.state = State
        self.app = self.compile_graph()

    # ─────────────── ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ───────────────
    def _log_end_time(self, state: State) -> None:
        """Логирует время выполнения запроса.

        Args:
            state: Текущее состояние графа.
        """
        end_time = time.time()
        duration = end_time - state["start_time"]
        print(f"⏱ Время выполнения: {duration:.2f}с")

    def _get_current_df(self) -> pd.DataFrame:
        """Загружает текущий DataFrame из файла.

        Если файл не найден, возвращает исходный self.df.

        Returns:
            Текущий DataFrame.
        """
        path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
        return self.df

    def _handle_block(self, block: str) -> tuple:
        """Парсит блок SSE-события.

        Args:
            block: Текст блока SSE.

        Returns:
            Кортеж (event_name, payload_dict).
        """
        event = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
        data_text = "\n".join(data_lines)
        payload = json.loads(data_text) if data_text else {}
        return event, payload

    def _listen_bytewise(self, job_id: str, timeout_seconds: int = 3000) -> None:
        """Прослушивает SSE-поток внешнего сервиса вычислений.

        Args:
            job_id: Идентификатор задачи на сервисе.
            timeout_seconds: Максимальное время ожидания.
        """
        url = f"{self.compute_tool_url}/events/{job_id}"
        with requests.get(
                url, stream=True, timeout=(1, None), headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            raw = resp.raw
            raw.decode_content = True
            buf = ""
            start = time.time()
            while True:
                chunk = raw.read(1)
                if not chunk:
                    print(CONNECTION_CLOSED_BY_SERVER)
                    break
                ch = chunk.decode("utf-8", errors="replace")
                buf += ch
                if buf.endswith("\n\n"):
                    block = buf.strip("\n")
                    buf = ""
                    event, payload = self._handle_block(block)
                    if event:
                        print(EVENT, event)
                    if event == "result" or payload.get("type") == "result":
                        print(HAS_RESULT)
                        result_url = payload.get("result_url")
                        print(RESULT_URL, result_url)
                        if result_url:
                            r2 = requests.get(self.compute_tool_url + result_url)
                            r2.raise_for_status()
                            result_path = os.path.join(
                                DATAFRAMES_DIR, f"result_{job_id}.pkl"
                            )
                            with open(result_path, "wb") as f:
                                f.write(r2.content)
                            print(SAVED_RESULT)
                            result_df = pd.read_pickle(result_path)
                            print(result_df.head(2))
                        break
                    if time.time() - start > timeout_seconds:
                        print(TIMEOUT)
                        break

    # ─────────────── ВЫЗОВЫ СУБ-АГЕНТОВ ───────────────
    def _invoke_code_graph_agent(
            self, data_structure: str, user_input: str
    ) -> str:
        """Вызывает Main Agent для генерации Python/Plotly кода.

        Форматирует системный промпт, подставляя структуру данных, текущую дату и день недели.
        Передаёт инструкцию использовать переменную ``df`` из sandbox.

        Args:
            data_structure: XML-представление head(2) текущего DataFrame.
            user_input: Запрос пользователя (с историей).

        Returns:
            Текстовый ответ агента или INVOKE_AGENT_LLM_ERROR.
        """
        days_ru = [
            "понедельник",
            "вторник",
            "среда",
            "четверг",
            "пятница",
            "суббота",
            "воскресенье",
        ]
        today = datetime.now().date()
        weekday = days_ru[datetime.today().weekday()]

        # Формируем системный промпт с подстановкой переменных
        system_prompt = prompt_react
        if "{data_structure}" in system_prompt:
            system_prompt = system_prompt.replace("{data_structure}", str(data_structure))
        else:
            system_prompt += f"\n\n<data_structure>\n{data_structure}\n</data_structure>"

        if "{today}" in system_prompt:
            system_prompt = system_prompt.replace("{today}", str(today))
        else:
            system_prompt += f"\nСегодня: {today}"

        if "{weekday}" in system_prompt:
            system_prompt = system_prompt.replace("{weekday}", str(weekday))
        else:
            system_prompt += f", день недели: {weekday}"

        system_prompt += (
            "\n\nВАЖНО: В среде выполнения доступна переменная `df` — это pandas "
            "DataFrame с данными. Используй её в генерируемом коде. "
            "Результат вычислений сохраняй в переменную, указанную как target_variable."
        )

        agent = get_react_main_agent(system_prompt)
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=user_input)],
            })
            print("## Сообщения Main react ##")
            for m in response["messages"]:
                print(m.content)
            print('')
            print('-' * 10)
            answer: str = response["messages"][-1].content
        except Exception as e:
            print(f"_invoke_code_graph_agent error: {e}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    def _invoke_stat_agent(self, data_source: str, user_input: str) -> str:
        """Вызывает Stat Agent для расчёта статистики по столбцам.

        Args:
            data_source: XML-представление head(2) текущего DataFrame.
            user_input: Запрос пользователя.

        Returns:
            Текстовый ответ агента или INVOKE_AGENT_LLM_ERROR.
        """
        system_prompt = prompt_react_stats
        if "{data_source}" in system_prompt:
            system_prompt = system_prompt.replace("{data_source}", str(data_source))
        else:
            system_prompt += f"\n\n<data_source>\n{data_source}\n</data_source>"

        agent = get_react_stat_agent(system_prompt)
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=user_input)],
            })
            answer: str = response["messages"][-1].content
        except Exception as e:
            print(f"_invoke_stat_agent error: {e}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    def _invoke_purpose_agent(self, user_input: str) -> str:
        """Вызывает Purpose Agent общего назначения.

        Args:
            user_input: Запрос пользователя.

        Returns:
            Текстовый ответ агента или INVOKE_AGENT_LLM_ERROR.
        """
        agent = get_react_purpose_agent(prompt_varios_purpose_agent)
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=user_input)],
            })
            answer: str = response["messages"][-1].content
        except Exception as e:
            print(f"_invoke_purpose_agent error: {e}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    # ─────────────── УЗЛЫ ГРАФА ───────────────
    def add_inform_in_user_input(self, state: State) -> Command:
        """Перефразирует запрос пользователя с учётом истории.

        Если история содержит >1 сообщения пользователя, вызывает LLM для перефразирования с учётом контекста.
        Если активна цепочка вычислений — сразу переходит к роутеру.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command с переходом на следующий узел.
        """
        start_time = time.time()
        if state.get("active_calculate_chain", False):
            return Command(
                goto="calculate_new_value_chain_router",
                update={"start_time": start_time},
            )

        if state["user_input"].count("##USER##:") > 1:
            prompt = ChatPromptTemplate.from_messages([
                ("system", promt_add_inform_in_query),
                ("human", "{user_input}"),
            ])
            chain = prompt | GigaChat_Pro
            response = chain.invoke({
                "user_input": state["user_input"][-MAX_CONTEXT_LENGTH:]
            })
            new_input = response.content

            last_user_position = state["user_input"].rfind("##USER##")
            ai_matches = re.findall(
                r"##AI AGENT##(.*?)(?=##|$)", state["user_input"], re.DOTALL
            )
            last_ai = ai_matches[-1].strip() if ai_matches else ""
            ai_message = f"##AI AGENT## {last_ai}"

            if last_user_position != -1:
                result = ai_message + " ##USER## " + new_input
            else:
                result = state["user_input"]

            return Command(
                goto="checking_for_common_request",
                update={"user_input": result, "start_time": start_time},
            )

        return Command(
            goto="checking_for_common_request",
            update={"start_time": start_time},
        )

    def checking_for_common_request(self, state: State) -> Command:
        """Классифицирует запрос: общий / выборка данных / новые вычисления.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command с переходом на соответствующий узел.
        """
        check_df = self._get_current_df()
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                    prompt_define_type_user_input +
                    "<data_structure>" +
                    check_df.head(2).to_xml() +
                    "</data_structure>" +
                    "<data_types>" +
                    ", ".join(f"{col}: {dtype}" for col, dtype in check_df.dtypes.items()) +
                    "</data_types>"
            )),
            ("human", "{user_input}"),
        ])
        chain = prompt | GigaChat_Pro
        response = chain.invoke({"user_input": state["user_input"]})
        content = response.content.lower().strip()

        if "общий" in content:
            return Command(goto="user_common_request")
        elif "выборка" in content:
            return Command(goto="cheking_that_query_belongs_table")
        elif "вычислен" in content:
            if not state.get("active_calculate_chain", False):
                return Command(
                    goto="query_doesnt_relate_table",
                    update={
                        "current_step_in_calculate_chain": 1,
                        "active_calculate_chain": True,
                    },
                )
            return Command(goto="calculate_new_value_chain_router")

        return Command(goto="user_common_request")

    def user_common_request(self, state: State) -> Command:
        """Обрабатывает общий запрос через Purpose Agent.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → END с final_answer.
        """
        answer = self._invoke_purpose_agent(state["user_input"])
        self._log_end_time(state)
        return Command(goto=END, update={"final_answer": answer})

    def calculate_new_value_chain_router(self, state: State) -> Command:
        """Маршрутизирует по шагам цепочки внешних вычислений.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command к нужному шагу или END.
        """
        step = state.get("current_step_in_calculate_chain", 0)
        if step == 1:
            return Command(goto="new_calculate_value_step_1_yes_no")
        elif step == 2:
            return Command(goto="new_calculate_value_step_2_post_job")
        return Command(goto=END)

    def new_calculate_value_step_1_yes_no(self, state: State) -> Command:
        """Шаг 1: пользователь подтверждает или отменяет вычисление.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → END с вопросом или отменой.
        """
        last_user_position = state["user_input"].rfind("##USER##")
        last_input = state["user_input"][last_user_position + len("##USER##"):]
        self._log_end_time(state)

        if "да" in last_input.lower():
            return Command(goto=END, update={
                "final_answer": (
                    "Введите описание вычисления. Это могут быть "
                    "few-shot примеры или описание алгоритма."
                ),
                "current_step_in_calculate_chain": 2,
            })
        else:
            return Command(goto=END, update={
                "final_answer": "Отмена вычисления нового значения",
                "current_step_in_calculate_chain": 0,
                "active_calculate_chain": False,
            })

    def new_calculate_value_step_2_post_job(self, state: State) -> Command:
        """Шаг 2: отправляет данные и описание на внешний сервис.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → END с job_id.
        """
        file_name = str(uuid.uuid4())
        pkl_path = os.path.join(DATAFRAMES_DIR, f"{file_name}.pkl")

        # Отправляем текущий DataFrame, а не исходный
        current_df = state.get("last_change_df", self.df)
        current_df.to_pickle(pkl_path)

        user_input = state["user_input"]
        # Извлекаем последние 3 сообщения пользователя
        parts = re.split(r"(##USER##|##AI AGENT##)", user_input)
        result = []
        current_label = None
        for part in parts:
            if part in ("##USER##", "##AI AGENT##"):
                current_label = part
            elif current_label:
                result.append((current_label, part.strip()))

        user_parts = [text for label, text in result if label == "##USER##" and text]
        last_3_user = user_parts[-3:]
        final_text = " ##USER##: ".join(last_3_user)

        with open(pkl_path, "rb") as f:
            files = {"file": (f"{file_name}.pkl", f, "application/octet-stream")}
            data = {"user_input": final_text}
            r = requests.post(
                f"{self.compute_tool_url}/jobs", files=files, data=data
            )
            r.raise_for_status()
            resp = r.json()
            job_id = resp["job_id"]

        t = threading.Thread(
            target=self._listen_bytewise, args=(job_id,), daemon=True
        )
        t.start()

        self._log_end_time(state)
        return Command(goto=END, update={
            "final_answer": job_id,
            "current_step_in_calculate_chain": 0,
            "active_calculate_chain": False,
        })

    def cheking_that_query_belongs_table(self, state: State) -> Command:
        """Проверяет, относится ли запрос к таблице.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → к уточняющему вопросу или query_doesnt_relate_table.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_cheking_that_query_belongs_table),
            ("human", "{user_input}"),
        ])
        chain = prompt | GigaChat_Max
        response = chain.invoke({"user_input": state["user_input"]})

        if response.content.strip().lower() in ("да", "yes", "true"):
            return Command(goto="cheking_need_make_additional_question")
        else:
            return Command(goto="query_doesnt_relate_table")

    def query_doesnt_relate_table(self, state: State) -> Command:
        """Возвращает сообщение о невозможности обработать запрос.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → END.
        """
        self._log_end_time(state)
        return Command(goto=END, update={"final_answer": QUERY_DOESNT_RELATE_TABLE})

    def cheking_need_make_additional_question(self, state: State) -> Command:
        """Проверяет необходимость уточняющего вопроса.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → make_additional_question или re_act_agent.
        """
        check_df = self._get_current_df()
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                    prompt_need_make_additional_question +
                    "<data_structure>" +
                    check_df.head(2).to_xml() +
                    "</data_structure>" +
                    "<data_types>" +
                    ", ".join(f"{col}: {dtype}" for col, dtype in check_df.dtypes.items()) +
                    "</data_types>"
            )),
            ("human", "{user_input}"),
        ])
        chain = prompt | GigaChat_Max
        response = chain.invoke({"user_input": state["user_input"]})

        if response.content.strip().lower() in ("да", "yes", "true"):
            return Command(goto="make_additional_question")
        else:
            return Command(goto="re_act_agent")

    def make_additional_question(self, state: State) -> Command:
        """Генерирует уточняющий вопрос для пользователя.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → END с текстом вопроса.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_make_additional_question),
            ("human", "{user_input}"),
        ])
        chain = prompt | GigaChat_Max
        response = chain.invoke({"user_input": state["user_input"]})
        return Command(goto=END, update={"final_answer": response.content})

    def re_act_agent(self, state: State) -> Command:
        # Получаем последний DataFrame из состояния или используем исходный
        last_change_df = state.get("last_change_df", self.df)

        # Сбрасываем песочницу, сохраняя базовые зависимости
        main_react_agent_sandbox.reset(keep_base=True)

        # Загружаем текущий DataFrame в песочницу
        load_dataframe_to_sandbox(main_react_agent_sandbox, last_change_df)

        # Преобразуем первые 2 строки DataFrame в XML для контекста
        last_change_df_xml = last_change_df.head(2).to_xml()

        # Вызываем граф-агент для обработки запроса пользователя
        answer = self._invoke_code_graph_agent(
            last_change_df_xml, state["user_input"]
        )

        # Проверяем тип результата из песочницы
        target_var = main_react_agent_sandbox.last_target_variable
        # print("TARGET VAR",target_var)
        if target_var is not None:
            target_val = main_react_agent_sandbox.get_variable(target_var)
            if target_val is not None:
                if isinstance(target_val, str) or isinstance(target_val, (float, int)):
                    answer = f"{answer}\n\nРезультат вычисления ({target_var}): {target_val}"
                    # print("Результат вычисления",target_var, type(target_val), target_val)
                    return Command(
                        goto="final_answer",
                        update={
                            "react_agent_answer": answer,
                            "last_change_df": last_change_df,  # остаётся прежним
                        },
                    )

        df_var = main_react_agent_sandbox.last_dataframe_variable
        # print("DF VAR REACT MAIN", df_var)
        if df_var is not None:
            val = main_react_agent_sandbox.get_variable(df_var)
            # print("DF VAR REACT MAIN VAL", val, type(val))
            if isinstance(val, (pd.DataFrame, pd.Series)) and not val.empty:
                if isinstance(val, pd.Series):
                    val = val.to_frame()
                # print("ЗАМЕНА DATAFRAME")
                last_change_df = val.copy()

        # Если результат не DataFrame, завершаем с текущим ответом
        if not isinstance(last_change_df, (pd.DataFrame, pd.Series)):
            return Command(
                goto="final_answer",
                update={"react_agent_answer": answer},
            )

        # Сохраняем текущий DataFrame
        current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
        last_change_df.to_pickle(current_path)

        # Сохраняем снапшот с именем на основе последнего ввода пользователя
        last_user_position = state["user_input"].rfind("##USER##")
        if last_user_position != -1:
            last_input = state["user_input"][last_user_position + len("##USER##"):]
            safe_name = re.sub(r"[^\w\s-]", "", last_input.strip())[:25].replace(" ", "_")
            # print("##SAFE NAME", safe_name)
            if safe_name:
                snapshot_path = os.path.join(DATAFRAMES_DIR, f"{safe_name}.pkl")
                last_change_df.to_pickle(snapshot_path)
                # print("## Сохранение в excel")
                last_change_df.to_excel(os.path.join(DATAFRAMES_DIR, f"{safe_name}.xlsx"))

        # print("last change df в react main:", last_change_df.head(1))
        return Command(
            goto="re_act_stat_agent",
            update={
                "react_agent_answer": answer,
                "last_change_df": last_change_df,
            },
        )

    def re_act_stat_agent(self, state: State) -> Command:
        """Вычисляет статистику по столбцам через Stat Agent.

        Чистит имена столбцов, пересохраняет current_dataframe.pkl и вызывает Stat Agent.

        Args:
            state: Текущее состояние графа.

        Returns:
            Command → final_answer.
        """
        last_change_df = state.get("last_change_df", self.df)

        if isinstance(last_change_df, pd.DataFrame) and len(last_change_df) > 3:
            # Чистим имена столбцов
            rename_map: dict = {}
            for col in last_change_df.columns:
                cleaned = re.sub(r"[^\w.-]", "_", str(col))
                if isinstance(col, int):
                    cleaned = f"col_{cleaned}"
                rename_map[col] = cleaned
            last_change_df = last_change_df.rename(columns=rename_map)

            # Пересохраняем с очищенными именами
            current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
            last_change_df.to_pickle(current_path)

            try:
                data_source = last_change_df.head(2).to_xml()
            except Exception:
                data_source = last_change_df.head(2).to_json()

            answer = self._invoke_stat_agent(data_source, state["user_input"])
        else:
            answer = ""

        return Command(
            goto="final_answer",
            update={
                "re_act_stat_agent_answer": answer,
                "last_change_df": last_change_df,  # Синхронизация state ↔ файл
            },
        )

    def final_answer(self, state: State) -> dict:
        """Формирует итоговый ответ на основе всех собранных данных.

        Объединяет: запрос пользователя, ответ Main Agent, ответ Stat Agent и head() DataFrame.

        Args:
            state: Текущее состояние графа.

        Returns:
            Словарь с ключом ``final_answer``.
        """
        last_change_df = state.get("last_change_df")
        prompt = ChatPromptTemplate.from_messages([
            ("system", FINAL_ANSWER_PROMPT),
            ("human", "{user_query}"),
        ])

        re_act_stat_agent_answer = state.get("re_act_stat_agent_answer", "")

        # Формируем XML из DataFrame
        try:
            if isinstance(last_change_df, pd.DataFrame) and not last_change_df.empty:
                last_change_df_xml = last_change_df.iloc[:20].to_xml()
            else:
                last_change_df_xml = "None"
        except Exception:
            last_change_df_xml = str(last_change_df) if isinstance(last_change_df, str) else "None"

        chain = prompt | GigaChat_Max

        print("запрос: ", state["user_input"])
        context = (
                          " Запрос пользователя: " + str(state["user_input"]) +
                          "Ответ Ai агента: " + state.get("react_agent_answer", "") +
                          "Ответ от Ai агента по вычислению статистику " + re_act_stat_agent_answer +
                          "Текущий набор данных был на обновлен на: " + last_change_df_xml +
                          "Это частичная выборка из набора данных, всего строк:" + str(len(last_change_df_xml))
                  )[-MAX_CONTEXT_LENGTH:]

        print("FINAL ANSWER CONTEXT:", context)
        try:
            response = chain.invoke({"user_query": context})
            content: str = response.content
        except Exception as e:
            print(f"Error in final_answer: {e}")
            content = INVOKE_AGENT_LLM_ERROR

        self._log_end_time(state)
        return {"final_answer": content}

    # ─────────────── КОМПИЛЯЦИЯ ГРАФА ───────────────
    def compile_graph(self):
        """Создаёт и компилирует StateGraph со всеми узлами и рёбрами.

        Returns:
            Скомпилированный граф LangGraph.
        """
        workflow = StateGraph(self.state)
        workflow.add_node("add_inform_in_user_input", self.add_inform_in_user_input)
        workflow.add_node("checking_for_common_request", self.checking_for_common_request)
        workflow.add_node("user_common_request", self.user_common_request)
        workflow.add_node("cheking_that_query_belongs_table", self.cheking_that_query_belongs_table)
        workflow.add_node("query_doesnt_relate_table", self.query_doesnt_relate_table)
        workflow.add_node("cheking_need_make_additional_question", self.cheking_need_make_additional_question)
        workflow.add_node("make_additional_question", self.make_additional_question)
        workflow.add_node("re_act_agent", self.re_act_agent)
        workflow.add_node("re_act_stat_agent", self.re_act_stat_agent)
        workflow.add_node("final_answer", self.final_answer)
        workflow.add_node("new_calculate_value_step_1_yes_no", self.new_calculate_value_step_1_yes_no)
        workflow.add_node("calculate_new_value_chain_router", self.calculate_new_value_chain_router)
        workflow.add_node("new_calculate_value_step_2_post_job", self.new_calculate_value_step_2_post_job)

        workflow.add_edge(START, "add_inform_in_user_input")
        workflow.add_edge("final_answer", END)

        return workflow.compile()


# ━━━━━━━━━━━━━━━━━━━ ТОЧКА ВХОДА ━━━━━━━━━━━━━━━━━━━
def start_agent(df: pd.DataFrame, compute_tool_url: str) -> None:
    """Запускает интерактивный цикл взаимодействия с агентом.

    Args:
        df: Исходный DataFrame для анализа.
        compute_tool_url: URL внешнего сервиса вычислений.
    """
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')

    HELLO_MSG = "⚙️: Начните взаимодействие с агентом"
    END_MSG = "⚙️: Завершение работы"
    QUIT_LIST = {"quit", "exit", "q", "й", "пока"}

    # Создаём директорию и сохраняем исходный DataFrame
    os.makedirs(DATAFRAMES_DIR, exist_ok=True)
    source_path = os.path.join(DATAFRAMES_DIR, "source_dataframe.pkl")
    current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
    df.to_pickle(source_path)
    df.to_pickle(current_path)

    messages: list[str] = []
    active_calculate_chain: bool = False
    current_step_in_calculate_chain: int = 0

    print(HELLO_MSG)
    while True:
        user_input = input("Ввод: ")
        if user_input.lower().strip() in QUIT_LIST:
            print(END_MSG)
            break

        messages.append(" ##USER##: " + user_input)
        agent = CSIAgent(df, compute_tool_url)

        # Загружаем текущий DataFrame из файла (учитывает изменения)
        if os.path.exists(current_path):
            current_df = pd.read_pickle(current_path)
        else:
            current_df = df

        final_answer = ""
        start_time = time.time()

        for chunk in agent.app.stream(
                {
                    "user_input": " ".join(messages[-7:]),
                    "active_calculate_chain": active_calculate_chain,
                    "current_step_in_calculate_chain": current_step_in_calculate_chain,
                    "start_time": start_time,
                    "last_change_df": current_df,
                },
                stream_mode="updates",
        ):
            for node_name, updates in chunk.items():
                print(f"\n🔵 Узел: [{node_name}]")
                if updates is None:
                    continue
                if "final_answer" in updates:
                    final_answer = updates["final_answer"]
                if "active_calculate_chain" in updates:
                    active_calculate_chain = updates["active_calculate_chain"]
                if "current_step_in_calculate_chain" in updates:
                    current_step_in_calculate_chain = updates["current_step_in_calculate_chain"]

        duration = time.time() - start_time
        messages.append(" ##AI AGENT##:" + final_answer)
        print(f"\n🤖 Ответ ({duration:.1f}с):\n{final_answer}")


if __name__ == "__main__":
    from load_data import get_data

    df = get_data()
    start_agent(df, "")