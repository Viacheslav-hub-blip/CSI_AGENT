from __future__ import annotations
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any
import pandas as pd
import requests
import urllib3
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict
from data_normalization_subgraph import normalization_service
from model import GigaChat_Max, GigaChat_Pro
from prompts import (
    prompt_cheking_that_query_belongs_table,
    prompt_define_type_user_input,
    prompt_make_additional_question,
    prompt_need_make_additional_question,
    prompt_react,
    prompt_react_stats,
    prompt_varios_purpose_agent,
    promt_add_inform_in_query,
)
from sub_agents import (
    get_react_main_agent,
    get_react_purpose_agent,
    get_react_stat_agent,
)
from tools import (
    DATAFRAMES_DIR,
    get_dataframe_from_sandbox,
    load_dataframe_to_sandbox,
    main_react_agent_sandbox,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
CURRENT_DATAFRAME_FILENAME = "current_dataframe.pkl"
SOURCE_DATAFRAME_FILENAME = "source_dataframe.pkl"
RESULT_FILENAME_TEMPLATE = "result_{job_id}.pkl"
USER_TAG = "##USER##"
AI_TAG = "##AI AGENT##"
USER_TAG_WITH_COLON = "##USER##:"
USER_HISTORY_PREFIX = " ##USER##: "
AI_HISTORY_PREFIX = " ##AI AGENT##: "
TAG_PATTERN = re.compile(r"(##USER##|##AI AGENT##)")
CONNECTION_CLOSED_BY_SERVER = "Соединение закрыто сервером"
EVENT = "Событие:"
HAS_RESULT = "Получен результат"
RESULT_URL = "URL результата:"
SAVED_RESULT = "Результат сохранен"
TIMEOUT = "Таймаут"
INVOKE_AGENT_LLM_ERROR = "Ошибка при вызове агента"
QUERY_DOESNT_RELATE_TABLE = "Запрос не относится к таблице"
FINAL_ANSWER_PROMPT = "Сформируйте итоговый ответ на основе контекста"
CALCULATE_CHAIN_PROMPT = "Введите описание вычисления. Это могут быть few-shot примеры или описание алгоритма."
CALCULATE_CHAIN_CANCEL_MESSAGE = "Отмена вычисления нового значения"
MAX_CONTEXT_LENGTH = 130_000
RECENT_USER_MESSAGES_FOR_JOB = 3
LISTENER_TIMEOUT_SECONDS = 3000
DATAFRAME_PREVIEW_ROWS = 2
FINAL_DATAFRAME_PREVIEW_ROWS = 20
SNAPSHOT_NAME_MAX_LENGTH = 25
SSE_REQUEST_TIMEOUT = (1, None)
SSE_HEADERS = {"Accept": "text/event-stream"}


class State(TypedDict):
    user_input: str
    last_change_df: Any
    react_agent_answer: str
    re_act_stat_agent_answer: str
    final_answer: str
    active_calculate_chain: bool
    active_normalization_chain: bool
    current_step_in_calculate_chain: int
    start_time: float


def _ensure_dataframes_dir() -> None:
    os.makedirs(DATAFRAMES_DIR, exist_ok=True)


def _current_dataframe_path() -> str:
    return os.path.join(DATAFRAMES_DIR, CURRENT_DATAFRAME_FILENAME)


def _source_dataframe_path() -> str:
    return os.path.join(DATAFRAMES_DIR, SOURCE_DATAFRAME_FILENAME)


def _extract_tagged_entries(history: str) -> list[tuple[str, str]]:
    if not history:
        return []
    parts = TAG_PATTERN.split(history)
    entries: list[tuple[str, str]] = []
    current_tag = ""
    for part in parts:
        if part in {USER_TAG, AI_TAG}:
            current_tag = part
            continue
        cleaned = part.strip(" :\n\t")
        if current_tag and cleaned:
            entries.append((current_tag, cleaned))
    return entries


def _extract_messages_by_tag(history: str, tag: str) -> list[str]:
    return [
        text
        for current_tag, text in _extract_tagged_entries(history)
        if current_tag == tag
    ]


def _extract_last_message_by_tag(history: str, tag: str) -> str:
    messages = _extract_messages_by_tag(history, tag)
    return messages[-1] if messages else ""


def _sanitize_snapshot_name(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text.strip())[:SNAPSHOT_NAME_MAX_LENGTH]
    return cleaned.replace(" ", "_")


def _format_dataframe_types(df: pd.DataFrame) -> str:
    return ", ".join(f"{column}: {dtype}" for column, dtype in df.dtypes.items())


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[Any, str] = {}
    for column in df.columns:
        cleaned = re.sub(r"[^\w.-]", "_", str(column))
        if isinstance(column, int):
            cleaned = f"col_{cleaned}"
        rename_map[column] = cleaned
    return df.rename(columns=rename_map)


def _dataframe_preview(df: Any, rows: int) -> str:
    if isinstance(df, pd.Series):
        preview_source = df.to_frame().head(rows)
    elif isinstance(df, pd.DataFrame):
        preview_source = df.head(rows)
    else:
        return "None"
    try:
        return preview_source.to_xml()
    except Exception:
        try:
            return preview_source.to_json()
        except Exception:
            return str(preview_source)


def _tail_text(value: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    return value[-max_length:]


def _escape_prompt_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


class CSIAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        compute_tool_url: str,
        thread_id: str | None = None,
        compute_job_callback: Any = None,
        compute_callback_url: str | None = None,
        compute_callback_thread_id: str | None = None,
    ) -> None:
        self.df = df
        self.thread_id = thread_id or "default"
        self.compute_tool_url = compute_tool_url.rstrip("/")
        self.compute_job_callback = compute_job_callback
        self.compute_callback_url = compute_callback_url
        self.compute_callback_thread_id = compute_callback_thread_id
        self.state = State
        self.app = self.compile_graph()

    def _normalization_session_id(self) -> str:
        return f"normalization::{self.thread_id}"

    def _log_end_time(self, state: State) -> None:
        start_time = state.get("start_time")
        if not start_time:
            return
        duration = time.time() - start_time
        print(f"Время выполнения: {duration:.2f}с")

    def _get_current_df(self) -> pd.DataFrame:
        current_path = _current_dataframe_path()
        if os.path.exists(current_path):
            return pd.read_pickle(current_path)
        return self.df

    def _persist_current_dataframe(self, df: pd.DataFrame) -> None:
        _ensure_dataframes_dir()
        df.to_pickle(_current_dataframe_path())

    def _persist_snapshot_dataframe(self, df: pd.DataFrame, user_input: str) -> None:
        _ensure_dataframes_dir()
        snapshot_name = _sanitize_snapshot_name(
            _extract_last_message_by_tag(user_input, USER_TAG)
        )
        if not snapshot_name:
            return
        snapshot_base = os.path.join(DATAFRAMES_DIR, snapshot_name)
        df.to_pickle(f"{snapshot_base}.pkl")
        try:
            df.to_excel(f"{snapshot_base}.xlsx", index=False)
        except Exception as exc:
            print(f"Could not export snapshot to xlsx: {exc}")

    def _build_dataframe_context(self, df: pd.DataFrame, prompt_text: str) -> str:
        data_structure = _dataframe_preview(df, DATAFRAME_PREVIEW_ROWS)
        data_types = _format_dataframe_types(df)
        result = prompt_text
        if "{data_structure}" in result:
            result = result.replace("{data_structure}", data_structure)
        else:
            result += f"<data_structure>{data_structure}</data_structure>"
        if "{data_types}" in result:
            result = result.replace("{data_types}", data_types)
        else:
            result += f"<data_types>{data_types}</data_types>"
        return result

    def _invoke_chat_prompt(self, system_prompt: str, user_input: str, llm: Any) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _escape_prompt_braces(system_prompt)),
                ("human", "{user_input}"),
            ]
        )
        chain = prompt | llm
        response = chain.invoke({"user_input": user_input})
        return getattr(response, "content", str(response))

    def _handle_block(self, block: str) -> tuple[str | None, dict[str, Any]]:
        event_name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        payload_text = "\n".join(data_lines)
        payload = json.loads(payload_text) if payload_text else {}
        return event_name, payload

    def _listen_bytewise(
        self,
        job_id: str,
        timeout_seconds: int = LISTENER_TIMEOUT_SECONDS,
    ) -> None:
        url = f"{self.compute_tool_url}/events/{job_id}"
        with requests.get(
            url,
            stream=True,
            timeout=SSE_REQUEST_TIMEOUT,
            headers=SSE_HEADERS,
        ) as response:
            response.raise_for_status()
            raw = response.raw
            raw.decode_content = True
            buffer = ""
            started_at = time.time()
            while True:
                chunk = raw.read(1)
                if not chunk:
                    print(CONNECTION_CLOSED_BY_SERVER)
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                if not buffer.endswith("\n\n"):
                    if time.time() - started_at > timeout_seconds:
                        print(TIMEOUT)
                        break
                    continue
                block = buffer.strip("\n")
                buffer = ""
                event_name, payload = self._handle_block(block)
                if event_name:
                    print(EVENT, event_name)
                if event_name == "result" or payload.get("type") == "result":
                    print(HAS_RESULT)
                    result_url = payload.get("result_url")
                    print(RESULT_URL, result_url)
                    if result_url:
                        result_response = requests.get(
                            f"{self.compute_tool_url}{result_url}", timeout=60
                        )
                        result_response.raise_for_status()
                        result_path = os.path.join(
                            DATAFRAMES_DIR,
                            RESULT_FILENAME_TEMPLATE.format(job_id=job_id),
                        )
                        with open(result_path, "wb") as file_obj:
                            file_obj.write(result_response.content)
                        print(SAVED_RESULT)
                    break
                if time.time() - started_at > timeout_seconds:
                    print(TIMEOUT)
                    break

    def _invoke_code_graph_agent(self, data_structure: str, user_input: str) -> str:
        today = datetime.now().date()
        weekdays_ru = [
            "понедельник",
            "вторник",
            "среда",
            "четверг",
            "пятница",
            "суббота",
            "воскресенье",
        ]
        weekday = weekdays_ru[datetime.today().weekday()]
        system_prompt = prompt_react
        if "{data_structure}" in system_prompt:
            system_prompt = system_prompt.replace("{data_structure}", data_structure)
        else:
            system_prompt += (
                f"\n\n<data_structure>\n{data_structure}\n</data_structure>"
            )
        if "{today}" in system_prompt:
            system_prompt = system_prompt.replace("{today}", str(today))
        else:
            system_prompt += f"\nСегодня: {today}"
        if "{weekday}" in system_prompt:
            system_prompt = system_prompt.replace("{weekday}", weekday)
        else:
            system_prompt += f"\nДень недели: {weekday}"
        system_prompt += (
            "\n\nВАЖНО: В среде выполнения доступна переменная `df` — это pandas "
            "DataFrame с данными. Используй ее в генерируемом коде. "
            "Результат вычислений сохраняй в переменную, указанную как target_variable."
        )
        agent = get_react_main_agent(system_prompt)
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            answer = response["messages"][-1].content
        except Exception as exc:
            print(f"_invoke_code_graph_agent error: {exc}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    def _invoke_stat_agent(self, data_source: str, user_input: str) -> str:
        system_prompt = prompt_react_stats
        if "{data_source}" in system_prompt:
            system_prompt = system_prompt.replace("{data_source}", data_source)
        else:
            system_prompt += f"\n\n<data_source>\n{data_source}\n</data_source>"
        agent = get_react_stat_agent(system_prompt)
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            answer = response["messages"][-1].content
        except Exception as exc:
            print(f"_invoke_stat_agent error: {exc}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    def _invoke_purpose_agent(self, user_input: str) -> str:
        agent = get_react_purpose_agent(prompt_varios_purpose_agent)
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            answer = response["messages"][-1].content
        except Exception as exc:
            print(f"_invoke_purpose_agent error: {exc}")
            answer = INVOKE_AGENT_LLM_ERROR
        return answer

    def _extract_job_user_input(self, history: str) -> str:
        user_messages = _extract_messages_by_tag(history, USER_TAG)
        return USER_HISTORY_PREFIX.join(user_messages[-RECENT_USER_MESSAGES_FOR_JOB:])

    def add_inform_in_user_input(self, state: State) -> Command:
        start_time = time.time()
        if state.get("active_calculate_chain", False):
            return Command(
                goto="calculate_new_value_chain_router",
                update={"start_time": start_time},
            )
        if state.get("active_normalization_chain", False):
            last_user_input = _extract_last_message_by_tag(
                state.get("user_input", ""), USER_TAG
            )
            if normalization_service.should_continue_session(
                self._normalization_session_id(),
                last_user_input,
            ):
                return Command(
                    goto="continue_normalization_chain",
                    update={"start_time": start_time},
                )
            return Command(
                goto="checking_for_common_request",
                update={"start_time": start_time},
            )
        user_input = state.get("user_input", "")
        if user_input.count(USER_TAG_WITH_COLON) <= 1:
            return Command(
                goto="checking_for_common_request",
                update={"start_time": start_time},
            )
        new_input = self._invoke_chat_prompt(
            system_prompt=promt_add_inform_in_query,
            user_input=_tail_text(user_input),
            llm=GigaChat_Pro,
        )
        last_ai_message = _extract_last_message_by_tag(user_input, AI_TAG)
        if USER_TAG in user_input:
            rebuilt_input = f"{AI_TAG} {last_ai_message} {USER_TAG} {new_input}".strip()
        else:
            rebuilt_input = user_input
        return Command(
            goto="checking_for_common_request",
            update={"user_input": rebuilt_input, "start_time": start_time},
        )

    def checking_for_common_request(self, state: State) -> Command:
        current_df = self._get_current_df()
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        if normalization_service.should_start_normalization(
            last_user_input, current_df
        ):
            return Command(goto="start_normalization_chain")
        prompt_text = self._build_dataframe_context(
            current_df, prompt_define_type_user_input
        )
        response = self._invoke_chat_prompt(
            prompt_text, state.get("user_input", ""), GigaChat_Pro
        )
        normalized_response = response.lower().strip()
        if "общий" in normalized_response:
            return Command(goto="user_common_request")
        if "выборка" in normalized_response:
            return Command(goto="cheking_that_query_belongs_table")
        if "вычислен" in normalized_response:
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
        answer = self._invoke_purpose_agent(state.get("user_input", ""))
        self._log_end_time(state)
        return Command(goto=END, update={"final_answer": answer})

    def start_normalization_chain(self, state: State) -> Command:
        current_df = self._get_current_df()
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        outcome = normalization_service.start_preview(
            self._normalization_session_id(),
            current_df,
            last_user_input,
        )
        self._log_end_time(state)
        return Command(
            goto=END,
            update={
                "final_answer": outcome.message,
                "active_normalization_chain": outcome.status
                in {"preview", "preview_updated"},
            },
        )

    def continue_normalization_chain(self, state: State) -> Command:
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        outcome = normalization_service.handle_reply(
            self._normalization_session_id(),
            last_user_input,
        )
        update: dict[str, Any] = {
            "final_answer": outcome.message,
            "active_normalization_chain": outcome.status
            in {"preview", "preview_updated"},
        }
        if outcome.status == "committed" and isinstance(
            outcome.dataframe, pd.DataFrame
        ):
            committed_df = outcome.dataframe.copy()
            update["final_answer"] = "Нормализация применена."
            self._persist_current_dataframe(committed_df)
            self._persist_snapshot_dataframe(
                committed_df,
                f"{USER_TAG_WITH_COLON} normalization_result",
            )
            update["last_change_df"] = committed_df
        self._log_end_time(state)
        return Command(goto=END, update=update)

    def calculate_new_value_chain_router(self, state: State) -> Command:
        step = state.get("current_step_in_calculate_chain", 0)
        if step == 1:
            return Command(goto="new_calculate_value_step_1_yes_no")
        if step == 2:
            return Command(goto="new_calculate_value_step_2_post_job")
        return Command(goto=END)

    def new_calculate_value_step_1_yes_no(self, state: State) -> Command:
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        self._log_end_time(state)
        if "да" in last_user_input.lower():
            return Command(
                goto=END,
                update={
                    "final_answer": CALCULATE_CHAIN_PROMPT,
                    "current_step_in_calculate_chain": 2,
                },
            )
        return Command(
            goto=END,
            update={
                "final_answer": CALCULATE_CHAIN_CANCEL_MESSAGE,
                "current_step_in_calculate_chain": 0,
                "active_calculate_chain": False,
            },
        )

    def new_calculate_value_step_2_post_job(self, state: State) -> Command:
        _ensure_dataframes_dir()
        file_name = str(uuid.uuid4())
        payload_path = os.path.join(DATAFRAMES_DIR, f"{file_name}.pkl")
        current_df = state.get("last_change_df", self.df)
        current_df.to_pickle(payload_path)
        with open(payload_path, "rb") as file_obj:
            files = {"file": (f"{file_name}.pkl", file_obj, "application/octet-stream")}
            data = {
                "user_input": self._extract_job_user_input(state.get("user_input", ""))
            }
            if self.compute_callback_url and self.compute_callback_thread_id:
                data["callback_url"] = self.compute_callback_url
                data["callback_thread_id"] = self.compute_callback_thread_id
            response = requests.post(
                f"{self.compute_tool_url}/jobs", files=files, data=data
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]
        if self.compute_job_callback is not None:
            self.compute_job_callback(job_id)
        elif not self.compute_callback_url:
            listener = threading.Thread(
                target=self._listen_bytewise,
                args=(job_id,),
                daemon=True,
            )
            listener.start()
        self._log_end_time(state)
        return Command(
            goto=END,
            update={
                "final_answer": f"Вычисление нового значения запущено в фоне. Идентификатор задачи: {job_id}",
                "current_step_in_calculate_chain": 0,
                "active_calculate_chain": False,
            },
        )

    def _legacy_cheking_that_query_belongs_table(self, state: State) -> Command:
        current_df = self._get_current_df()
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        prompt_text = self._build_dataframe_context(
            current_df,
            prompt_cheking_that_query_belongs_table
            + "\nУчитывай семантическое соответствие между запросом и названиями столбцов."
            + "\nСчитай совпадением перевод, синонимы, склонения и близкие формулировки."
            + "\nНапример, `страны` может соответствовать столбцу `country`, а `население` — `Population`.",
        )
        response = self._invoke_chat_prompt(
            prompt_text,
            last_user_input or state.get("user_input", ""),
            GigaChat_Max,
        )
        if response.strip().lower() in ("РґР°", "yes", "true"):
            return Command(goto="cheking_need_make_additional_question")
        return Command(goto="query_doesnt_relate_table")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_cheking_that_query_belongs_table),
                ("human", "{user_input}"),
            ]
        )
        chain = prompt | GigaChat_Max
        response = chain.invoke({"user_input": state.get("user_input", "")})
        if response.content.strip().lower() in ("да", "yes", "true"):
            return Command(goto="cheking_need_make_additional_question")
        return Command(goto="query_doesnt_relate_table")

    def query_doesnt_relate_table(self, state: State) -> Command:
        self._log_end_time(state)
        return Command(goto=END, update={"final_answer": QUERY_DOESNT_RELATE_TABLE})

    def cheking_that_query_belongs_table(self, state: State) -> Command:
        current_df = self._get_current_df()
        last_user_input = _extract_last_message_by_tag(
            state.get("user_input", ""), USER_TAG
        )
        prompt_text = self._build_dataframe_context(
            current_df,
            prompt_cheking_that_query_belongs_table
            + "\nУчитывай семантическое соответствие между запросом и названиями столбцов."
            + "\nСчитай совпадением перевод, синонимы, склонения и близкие формулировки."
            + "\nНапример, `страны` может соответствовать столбцу `country`, а `население` — `Population`.",
        )
        response = self._invoke_chat_prompt(
            prompt_text,
            last_user_input or state.get("user_input", ""),
            GigaChat_Max,
        )
        if response.strip().lower() in ("да", "yes", "true"):
            return Command(goto="cheking_need_make_additional_question")
        return Command(goto="query_doesnt_relate_table")

    def cheking_need_make_additional_question(self, state: State) -> Command:
        current_df = self._get_current_df()
        prompt_text = self._build_dataframe_context(
            current_df,
            prompt_need_make_additional_question,
        )
        response = self._invoke_chat_prompt(
            prompt_text, state.get("user_input", ""), GigaChat_Max
        )
        if response.strip().lower() in ("да", "yes", "true"):
            return Command(goto="make_additional_question")
        return Command(goto="re_act_agent")

    def make_additional_question(self, state: State) -> Command:
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_make_additional_question), ("human", "{user_input}")]
        )
        chain = prompt | GigaChat_Max
        response = chain.invoke({"user_input": state.get("user_input", "")})
        return Command(goto=END, update={"final_answer": response.content})

    def re_act_agent(self, state: State) -> Command:
        last_change_df = state.get("last_change_df", self.df)
        main_react_agent_sandbox.reset(keep_base=True)
        load_dataframe_to_sandbox(main_react_agent_sandbox, last_change_df)
        data_structure = _dataframe_preview(last_change_df, DATAFRAME_PREVIEW_ROWS)
        answer = self._invoke_code_graph_agent(
            data_structure, state.get("user_input", "")
        )
        target_variable = getattr(
            main_react_agent_sandbox, "last_target_variable", None
        )
        if target_variable:
            target_value = main_react_agent_sandbox.get_variable(target_variable)
            if isinstance(target_value, (str, float, int)):
                answer = f"{answer}\n\nРезультат вычисления ({target_variable}): {target_value}"
                return Command(
                    goto="final_answer",
                    update={
                        "react_agent_answer": answer,
                        "last_change_df": last_change_df,
                    },
                )
        dataframe_result = get_dataframe_from_sandbox(main_react_agent_sandbox)
        if isinstance(dataframe_result, pd.Series):
            dataframe_result = dataframe_result.to_frame()
        if isinstance(dataframe_result, pd.DataFrame) and not dataframe_result.empty:
            last_change_df = dataframe_result.copy()
        if not isinstance(last_change_df, pd.DataFrame):
            return Command(goto="final_answer", update={"react_agent_answer": answer})
        self._persist_current_dataframe(last_change_df)
        self._persist_snapshot_dataframe(last_change_df, state.get("user_input", ""))
        return Command(
            goto="re_act_stat_agent",
            update={
                "react_agent_answer": answer,
                "last_change_df": last_change_df,
            },
        )

    def re_act_stat_agent(self, state: State) -> Command:
        last_change_df = state.get("last_change_df", self.df)
        if isinstance(last_change_df, pd.DataFrame) and len(last_change_df) > 3:
            last_change_df = _normalize_dataframe_columns(last_change_df)
            self._persist_current_dataframe(last_change_df)
            data_source = _dataframe_preview(last_change_df, DATAFRAME_PREVIEW_ROWS)
            answer = self._invoke_stat_agent(data_source, state.get("user_input", ""))
        else:
            answer = ""
        return Command(
            goto="final_answer",
            update={
                "re_act_stat_agent_answer": answer,
                "last_change_df": last_change_df,
            },
        )

    def final_answer(self, state: State) -> dict[str, Any]:
        last_change_df = state.get("last_change_df")
        dataframe_preview = _dataframe_preview(
            last_change_df, FINAL_DATAFRAME_PREVIEW_ROWS
        )
        context = (
            "Запрос пользователя: "
            + str(state.get("user_input", ""))
            + " Ответ AI агента: "
            + state.get("react_agent_answer", "")
            + " Ответ AI агента по вычислению статистики: "
            + state.get("re_act_stat_agent_answer", "")
            + " Текущий набор данных обновлен до: "
            + dataframe_preview
        )[-MAX_CONTEXT_LENGTH:]
        prompt = ChatPromptTemplate.from_messages(
            [("system", FINAL_ANSWER_PROMPT), ("human", "{user_query}")]
        )
        chain = prompt | GigaChat_Max
        try:
            response = chain.invoke({"user_query": context})
            content = response.content
        except Exception as exc:
            print(f"Error in final_answer: {exc}")
            content = INVOKE_AGENT_LLM_ERROR
        self._log_end_time(state)
        return {"final_answer": content}

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("add_inform_in_user_input", self.add_inform_in_user_input)
        workflow.add_node(
            "checking_for_common_request", self.checking_for_common_request
        )
        workflow.add_node("user_common_request", self.user_common_request)
        workflow.add_node("start_normalization_chain", self.start_normalization_chain)
        workflow.add_node(
            "continue_normalization_chain", self.continue_normalization_chain
        )
        workflow.add_node(
            "cheking_that_query_belongs_table", self.cheking_that_query_belongs_table
        )
        workflow.add_node("query_doesnt_relate_table", self.query_doesnt_relate_table)
        workflow.add_node(
            "cheking_need_make_additional_question",
            self.cheking_need_make_additional_question,
        )
        workflow.add_node("make_additional_question", self.make_additional_question)
        workflow.add_node("re_act_agent", self.re_act_agent)
        workflow.add_node("re_act_stat_agent", self.re_act_stat_agent)
        workflow.add_node("final_answer", self.final_answer)
        workflow.add_node(
            "new_calculate_value_step_1_yes_no", self.new_calculate_value_step_1_yes_no
        )
        workflow.add_node(
            "calculate_new_value_chain_router", self.calculate_new_value_chain_router
        )
        workflow.add_node(
            "new_calculate_value_step_2_post_job",
            self.new_calculate_value_step_2_post_job,
        )
        workflow.add_edge(START, "add_inform_in_user_input")
        workflow.add_edge("final_answer", END)
        return workflow.compile()


def start_agent(df: pd.DataFrame, compute_tool_url: str) -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    _ensure_dataframes_dir()
    df.to_pickle(_source_dataframe_path())
    df.to_pickle(_current_dataframe_path())
    messages: list[str] = []
    active_calculate_chain = False
    active_normalization_chain = False
    current_step_in_calculate_chain = 0
    print("Начните взаимодействие с агентом")
    while True:
        user_input = input("Ввод: ").strip()
        if user_input.lower() in {"quit", "exit", "q", "й", "пока"}:
            print("Завершение работы")
            break
        messages.append(USER_HISTORY_PREFIX + user_input)
        agent = CSIAgent(df, compute_tool_url, thread_id="cli_session")
        current_df = (
            pd.read_pickle(_current_dataframe_path())
            if os.path.exists(_current_dataframe_path())
            else df
        )
        state: State = {
            "user_input": " ".join(messages),
            "active_calculate_chain": active_calculate_chain,
            "active_normalization_chain": active_normalization_chain,
            "current_step_in_calculate_chain": current_step_in_calculate_chain,
            "start_time": time.time(),
            "last_change_df": current_df,
            "react_agent_answer": "",
            "re_act_stat_agent_answer": "",
            "final_answer": "",
        }
        final_answer = ""
        for chunk in agent.app.stream(state, stream_mode="updates"):
            for _, updates in chunk.items():
                if "final_answer" in updates:
                    final_answer = updates["final_answer"]
                if "active_calculate_chain" in updates:
                    active_calculate_chain = updates["active_calculate_chain"]
                if "active_normalization_chain" in updates:
                    active_normalization_chain = updates["active_normalization_chain"]
                if "current_step_in_calculate_chain" in updates:
                    current_step_in_calculate_chain = updates[
                        "current_step_in_calculate_chain"
                    ]
        messages.append(AI_HISTORY_PREFIX + final_answer)
        print(final_answer)
