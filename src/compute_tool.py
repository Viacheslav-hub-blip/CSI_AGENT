import ast
import asyncio
import contextlib
import io
import json
import logging
import traceback
from typing import Any, Callable, Dict, TypedDict
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from prompts_compute_tool import (
    prompt_determine_need_llm,
    prompt_for_generate_prompt,
    prompt_generate_python_func_with_giga,
    prompt_generate_python_func_without_giga,
)

logger = logging.getLogger(__name__)
MAX_RETRIES = 3
ProgressReporter = Callable[[int, int, str], None]
StatusReporter = Callable[[str], None]
ASYNC_ROW_CONCURRENCY_LIMIT = 5
ROW_PROGRESS_UPDATE_EVERY = 5


def _escape_prompt_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _safe_dataframe_preview(df: pd.DataFrame, rows: int = 1) -> str:
    preview_df = df.head(rows)
    try:
        return preview_df.to_xml()
    except Exception:
        try:
            return preview_df.to_json()
        except Exception:
            return str(preview_df)


class State(TypedDict, total=False):
    user_input: str
    generated_prompt: str
    generated_python_code_with_llm: str
    generated_python_code_without_llm: str
    code_exec_error: str
    code_exec_stdout: str
    attempt: int
    result_status: str


class AsyncNamespaceShell:
    def __init__(self) -> None:
        self.user_ns: dict[str, Any] = {}


def _emit_status_message(
    message: str, status_reporter: StatusReporter | None = None
) -> None:
    normalized_message = str(message or "").strip()
    if not normalized_message:
        return
    print(normalized_message, flush=True)
    if status_reporter is not None:
        status_reporter(normalized_message)


def _build_retry_feedback(state: State, code_key: str) -> dict[str, str | int]:
    return {
        "executed_code": state.get(code_key, "Code was not generated yet"),
        "error_from_previous_step": state.get(
            "code_exec_error", "Code was not executed yet"
        ),
        "stdout_from_previous_step": state.get(
            "code_exec_stdout", "No stdout from previous run."
        ),
        "retry_attempt": state.get("attempt", 0),
    }


async def _gather_with_concurrency_limit(
    coroutines: list[Any],
    limit: int,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run_single(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*[_run_single(coroutine) for coroutine in coroutines])


def run_async_tasks_limited(
    coroutines: list[Any],
    limit: int = ASYNC_ROW_CONCURRENCY_LIMIT,
) -> list[Any]:
    return asyncio.run(_gather_with_concurrency_limit(list(coroutines), limit))


def async_process_rows_limited(
    rows: list[Any],
    process_row_async: Callable[[Any], Any],
    limit: int = ASYNC_ROW_CONCURRENCY_LIMIT,
    progress_reporter: ProgressReporter | None = None,
    progress_message: str = "Идет асинхронная обработка строк",
) -> list[Any]:
    rows_list = list(rows)
    total_rows = len(rows_list)
    update_every = max(1, min(ROW_PROGRESS_UPDATE_EVERY, total_rows or 1))
    if progress_reporter is not None:
        progress_reporter(0, total_rows, progress_message)

    async def _runner() -> list[Any]:
        semaphore = asyncio.Semaphore(max(1, limit))
        processed_rows = 0

        async def _run_single(row: Any) -> Any:
            nonlocal processed_rows
            async with semaphore:
                result = await process_row_async(row)
            processed_rows += 1
            if progress_reporter is not None and (
                processed_rows == total_rows or processed_rows % update_every == 0
            ):
                progress_reporter(
                    processed_rows,
                    total_rows,
                    f"{progress_message}: {processed_rows}/{total_rows}",
                )
            return result

        return await asyncio.gather(*[_run_single(row) for row in rows_list])

    return asyncio.run(_runner())


class MyRunner:
    def __init__(self, shell: AsyncNamespaceShell | None = None) -> None:
        self.shell = shell or AsyncNamespaceShell()

    @staticmethod
    def _execute_code(
        shell: AsyncNamespaceShell, code: str
    ) -> tuple[str, str, str, Any]:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        exec_result = None
        error_text = ""
        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
            ):
                exec(code, shell.user_ns, shell.user_ns)
        except Exception:
            error_text = traceback.format_exc()
            err_buf.write(error_text)
        return out_buf.getvalue(), err_buf.getvalue(), error_text, exec_result

    async def run_code(
        self,
        code: str,
        source_dataframe: pd.DataFrame,
        prompt: str,
        GigaChat_Max,
        show_in_notebook: bool = False,
        progress_reporter: ProgressReporter | None = None,
        status_reporter: StatusReporter | None = None,
    ) -> Dict[str, Any]:
        del show_in_notebook
        logger.info("##RUN CODE##")
        shell = self.shell
        total_rows = len(source_dataframe)
        shell.user_ns.update(
            {
                "pd": pd,
                "np": np,
                "json": json,
                "asyncio": asyncio,
                "source_dataframe": source_dataframe.copy(),
                "df": source_dataframe.copy(),
                "prompt": prompt,
                "GigaChat_Max": GigaChat_Max,
                "llm": GigaChat_Max,
                "report_progress": (
                    progress_reporter
                    if progress_reporter is not None
                    else (lambda *_args: None)
                ),
                "print_status": (
                    lambda message: _emit_status_message(message, status_reporter)
                ),
                "run_async_tasks_limited": run_async_tasks_limited,
                "async_process_rows_limited": (
                    lambda rows, process_row_async, limit=ASYNC_ROW_CONCURRENCY_LIMIT, progress_message="Идет асинхронная обработка строк": (
                        async_process_rows_limited(
                            rows=list(rows),
                            process_row_async=process_row_async,
                            limit=limit,
                            progress_reporter=progress_reporter,
                            progress_message=progress_message,
                        )
                    )
                ),
                "ASYNC_ROW_CONCURRENCY_LIMIT": ASYNC_ROW_CONCURRENCY_LIMIT,
            }
        )
        _emit_status_message(
            "Compute tool: подготовка выполнения кода", status_reporter
        )
        if progress_reporter is not None:
            progress_reporter(0, total_rows, "Выполнение кода запущено")
        stdout_text, stderr_raw, error_text, exec_result = await asyncio.to_thread(
            self._execute_code,
            shell,
            code,
        )
        stderr_text = stderr_raw
        logger.info(
            "##RUN CODE## stderr",
            extra={"extra_info": f"Logs: {stderr_text}"},
        )
        result_dataframe = None
        for candidate in ("final_df", "result_df", "df", "source_dataframe"):
            value = shell.user_ns.get(candidate)
            if isinstance(value, pd.DataFrame):
                result_dataframe = value.copy()
                break
        if progress_reporter is not None and error_text == "":
            progress_reporter(total_rows, total_rows, "Выполнение кода завершено")
        if error_text:
            _emit_status_message(
                "Compute tool: выполнение кода завершилось ошибкой", status_reporter
            )
        else:
            _emit_status_message(
                "Compute tool: выполнение кода завершено", status_reporter
            )
        return {
            "success": error_text == "",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exec_result": exec_result,
            "result_dataframe": result_dataframe,
        }


class ProcessDataFrameAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        llm_model,
        progress_reporter: ProgressReporter | None = None,
        status_reporter: StatusReporter | None = None,
    ) -> None:
        self.state = State
        self.df = df.copy()
        self.current_dataframe = df.copy()
        self.llm_model = llm_model
        self.progress_reporter = progress_reporter
        self.status_reporter = status_reporter
        self.python_executor = MyRunner()
        self.app = self.compile_graph()

    @staticmethod
    def _clean_code_block(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("python"):
                cleaned = cleaned[len("python") :].lstrip()
        return cleaned.strip()

    @staticmethod
    def _normalize_prompt_payload(raw_content: str) -> dict[str, str]:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[len("json") :].lstrip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = ast.literal_eval(cleaned)
        defaults = {
            "agent_role": "Data extraction assistant",
            "instructions": "Extract the requested value from the row and return only the value.",
            "important": "Do not invent facts. Return an empty string when the value is missing.",
            "input_format": "One dataframe row serialized to text.",
            "output_format": "A plain string with the extracted value.",
            "available_values": "Use only values present in the row.",
            "examples": "Input: row text. Output: extracted value.",
        }
        defaults.update({key: str(value) for key, value in payload.items()})
        return defaults

    async def _prompt_template(self, values: dict[str, str]) -> str:
        return f"""
<role>
    {values["agent_role"]}
</role>
<instructions>
    {values["instructions"]}
</instructions>
<important>
    {values["important"]}
</important>
<input_format>
    {values["input_format"]}
</input_format>
<output_format>
    {values["output_format"]}
</output_format>
<available_values>
    {values["available_values"]}
</available_values>
<examples>
    {values["examples"]}
</examples>
""".strip()

    async def determining_need_use_llm(self, state: State):
        logger.info("##determining_need_use_llm##")
        data_structure = _safe_dataframe_preview(self.df)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    _escape_prompt_braces(
                        prompt_determine_need_llm + "<data_structure>" + data_structure
                    ),
                ),
                ("human", "{user_query}"),
            ]
        )
        chain = prompt | self.llm_model
        response = await chain.ainvoke({"user_query": state["user_input"]})
        content = response.content.lower().strip()
        next_step = (
            "generate_prompt"
            if content in ("yes", "true", "да")
            else "generate_python_code_without_llm"
        )
        return Command(goto=next_step)

    async def generate_prompt(self, state: State):
        logger.info("##GENERATE_PROMPT##")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_for_generate_prompt),
                ("human", "{user_query}"),
            ]
        )
        chain = prompt | self.llm_model
        response = await chain.ainvoke({"user_query": state["user_input"]})
        response_dict = self._normalize_prompt_payload(response.content)
        prompt_for_parsing = await self._prompt_template(response_dict)
        logger.info(
            "##GENERATE_PROMPT## parsed",
            extra={"extra_info": prompt_for_parsing},
        )
        return Command(
            goto="generate_python_code_with_llm",
            update={"generated_prompt": prompt_for_parsing},
        )

    async def generate_python_code_with_llm(self, state: State):
        logger.info("##GENERATE_PYTHON_CODE_WITH_LLM##")
        retry_feedback = _build_retry_feedback(state, "generated_python_code_with_llm")
        system_prompt = prompt_generate_python_func_with_giga.replace(
            "data_structure_replaced",
            _safe_dataframe_preview(self.df),
        )
        prompt_python = ChatPromptTemplate.from_messages(
            [
                ("system", _escape_prompt_braces(system_prompt)),
                (
                    "human",
                    (
                        "User request: {user_query}. Retry attempt: {retry_attempt}. "
                        "Previous generated code: {executed_code}. "
                        "Previous stdout: {stdout_from_previous_step}. "
                        "Previous execution error: {error_from_previous_step}. "
                        "Prompt template for extraction: {generated_prompt}"
                    ),
                ),
            ]
        )
        chain_python = prompt_python | self.llm_model
        response = await chain_python.ainvoke(
            {
                "user_query": state["user_input"],
                "executed_code": retry_feedback["executed_code"],
                "error_from_previous_step": retry_feedback["error_from_previous_step"],
                "stdout_from_previous_step": retry_feedback[
                    "stdout_from_previous_step"
                ],
                "retry_attempt": retry_feedback["retry_attempt"],
                "generated_prompt": state.get("generated_prompt", ""),
            }
        )
        code = self._clean_code_block(response.content)
        return Command(
            goto="execute_python_code",
            update={"generated_python_code_with_llm": code},
        )

    async def generate_python_code_without_llm(self, state: State):
        logger.info("##GENERATE_PYTHON_CODE_WITHOUT_LLM##")
        retry_feedback = _build_retry_feedback(
            state, "generated_python_code_without_llm"
        )
        schema = (
            _safe_dataframe_preview(self.df)
            + "<data_types>"
            + ", ".join(f"{col}: {dtype}" for col, dtype in self.df.dtypes.items())
            + "</data_types>"
            + "<empty_values>"
            + ", ".join(
                f"{col}: {self.df[col].isna().any()}" for col in self.df.columns
            )
            + "</empty_values>"
        )
        system_prompt = prompt_generate_python_func_without_giga.replace(
            "data_structure_replaced",
            schema,
        )
        prompt_python = ChatPromptTemplate.from_messages(
            [
                ("system", _escape_prompt_braces(system_prompt)),
                (
                    "human",
                    (
                        "User request: {user_query}. Retry attempt: {retry_attempt}. "
                        "Previous generated code: {executed_code}. "
                        "Previous stdout: {stdout_from_previous_step}. "
                        "Previous execution error: {error_from_previous_step}."
                    ),
                ),
            ]
        )
        chain_python = prompt_python | self.llm_model
        response = await chain_python.ainvoke(
            {
                "user_query": state["user_input"],
                "executed_code": retry_feedback["executed_code"],
                "error_from_previous_step": retry_feedback["error_from_previous_step"],
                "stdout_from_previous_step": retry_feedback[
                    "stdout_from_previous_step"
                ],
                "retry_attempt": retry_feedback["retry_attempt"],
            }
        )
        code = self._clean_code_block(response.content)
        return Command(
            goto="execute_python_code",
            update={"generated_python_code_without_llm": code},
        )

    async def execute_python_code(self, state: State):
        logger.info("##EXECUTE_PYTHON_CODE##")
        code = state.get("generated_python_code_with_llm") or state.get(
            "generated_python_code_without_llm",
            "",
        )
        print("=== COMPUTE TOOL GENERATED CODE START ===", flush=True)
        print(code, flush=True)
        print("=== COMPUTE TOOL GENERATED CODE END ===", flush=True)
        generated_prompt = state.get("generated_prompt", "")
        exec_res = await self.python_executor.run_code(
            code,
            self.current_dataframe,
            generated_prompt,
            self.llm_model,
            progress_reporter=self.progress_reporter,
            status_reporter=self.status_reporter,
        )
        logger.info("##EXECUTE_PYTHON_CODE## result", extra={"extra_info": exec_res})
        if exec_res["success"]:
            if isinstance(exec_res.get("result_dataframe"), pd.DataFrame):
                self.current_dataframe = exec_res["result_dataframe"]
                self.df = exec_res["result_dataframe"]
            return {"result_status": "done"}
        print("=== COMPUTE TOOL EXECUTION ERROR START ===", flush=True)
        print(exec_res["stderr"], flush=True)
        print("=== COMPUTE TOOL EXECUTION ERROR END ===", flush=True)
        attempt = state.get("attempt", 0) + 1
        if attempt >= MAX_RETRIES:
            return {
                "result_status": "error",
                "code_exec_error": exec_res["stderr"],
                "code_exec_stdout": exec_res["stdout"],
                "attempt": attempt,
            }
        target_node = (
            "generate_python_code_with_llm"
            if state.get("generated_python_code_with_llm")
            else "generate_python_code_without_llm"
        )
        return Command(
            goto=target_node,
            update={
                "code_exec_error": exec_res["stderr"],
                "code_exec_stdout": exec_res["stdout"],
                "attempt": attempt,
            },
        )

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("determining_need_use_llm", self.determining_need_use_llm)
        workflow.add_node("generate_prompt", self.generate_prompt)
        workflow.add_node(
            "generate_python_code_with_llm", self.generate_python_code_with_llm
        )
        workflow.add_node(
            "generate_python_code_without_llm",
            self.generate_python_code_without_llm,
        )
        workflow.add_node("execute_python_code", self.execute_python_code)
        workflow.add_edge(START, "determining_need_use_llm")
        workflow.add_edge("execute_python_code", END)
        return workflow.compile()

    async def __call__(self, user_input: str):
        return await self.app.ainvoke({"user_input": user_input})

    @staticmethod
    def _extract_token_text(event: dict[str, Any]) -> str:
        data = event.get("data", {})
        chunk = data.get("chunk")
        if chunk is None:
            return ""
        if hasattr(chunk, "content"):
            return str(chunk.content)
        if isinstance(chunk, str):
            return chunk
        return ""

    async def run_with_streaming(self, inputs):
        last_step = ""
        generated_prompt_once = False
        async for event in self.app.astream_events(inputs):
            metadata = event.get("metadata") or {}
            graph_node = metadata.get("langgraph_node", "")
            if graph_node == "generate_prompt" and last_step != "generate_prompt":
                last_step = "generate_prompt"
                if not generated_prompt_once:
                    generated_prompt_once = True
                    yield "Generating extraction prompt..."
                else:
                    yield "Retrying prompt generation after execution error..."
            if (
                graph_node == "determining_need_use_llm"
                and last_step != "determining_need_use_llm"
            ):
                last_step = "determining_need_use_llm"
                yield "Checking whether the task needs an LLM-based extraction step..."
            if (
                graph_node == "generate_python_code_with_llm"
                and last_step != "generate_python_code_with_llm"
            ):
                last_step = "generate_python_code_with_llm"
                yield "Generating Python code with LLM support..."
            if (
                graph_node == "generate_python_code_without_llm"
                and last_step != "generate_python_code_without_llm"
            ):
                last_step = "generate_python_code_without_llm"
                yield "Generating deterministic Python code..."
            if (
                graph_node == "execute_python_code"
                and last_step != "execute_python_code"
            ):
                last_step = "execute_python_code"
                yield "Executing Python code..."
