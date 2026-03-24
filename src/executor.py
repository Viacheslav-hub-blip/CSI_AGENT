import json
import re
import asyncio
from typing import Optional, Any, Dict, Type
from pydantic import Field, ConfigDict, BaseModel, PrivateAttr, create_model
from langchain_core.tools import BaseTool
from sandbox import ClientPythonSandbox, ExecutionResult


class BaseCodeExecutorTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    args_schema: Optional[Type[BaseModel]] = None
    mcp_tool: Any = Field(default=None, exclude=True, repr=False)
    sandbox: ClientPythonSandbox = Field(default=None, exclude=True, repr=False)
    used_libraries: Optional[str] = Field(default=None, exclude=True, repr=False)

    # Состояние для retry-логики (Pydantic v2 PrivateAttr)

    _previous_code: Optional[str] = PrivateAttr(default=None)
    _error_context: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self,
        mcp_tool: Any,
        sandbox: ClientPythonSandbox,
        used_libraries: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        mcp_schema = getattr(mcp_tool, "args_schema", None)
        kwargs["args_schema"] = self._convert_schema(mcp_schema) if mcp_schema else None
        kwargs["mcp_tool"] = mcp_tool
        kwargs["sandbox"] = sandbox
        kwargs["used_libraries"] = used_libraries
        super().__init__(**kwargs)

    @staticmethod
    def _convert_schema(schema: Any) -> Optional[Type[BaseModel]]:
        if isinstance(schema, type):
            return schema
        if isinstance(schema, dict):
            return BaseCodeExecutorTool._json_schema_to_pydantic(schema)
        return None

    @staticmethod
    def _json_schema_to_pydantic(json_schema: dict) -> Type[BaseModel]:
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])
        TYPE_MAP = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        fields: Dict[str, Any] = {}
        for name, info in properties.items():
            has_default = "default" in info
            is_optional = has_default or name not in required
            if "anyOf" in info:
                is_optional = True

            # Определяем тип

            json_type = info.get("type", "string")
            python_type = TYPE_MAP.get(json_type, str)

            # Обрабатываем enum

            enum_values = info.get("enum")
            description = info.get("description", "")
            if enum_values:
                from enum import Enum

                enum_name = f"{name}_enum"
                enum_cls = Enum(enum_name, {v: v for v in enum_values})
                python_type = enum_cls
                description += f" Допустимые значения: {enum_values}"

            # Optional или required

            if is_optional:
                field_type = Optional[python_type]
                default = info.get("default", None)
            else:
                field_type = python_type
                default = ...
            fields[name] = (
                field_type,
                Field(default=default, description=description),
            )
        return create_model("DynamicMCPSchema", **fields)

    def _run(self, **kwargs: Any) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Внутри уже работающего event loop (Jupyter / async)

            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(self._arun(**kwargs))
        else:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        mcp_args = self._prepare_mcp_args(**kwargs)
        generated_code = await self._invoke_mcp_and_parse(mcp_args)
        if generated_code is None:
            return self._make_error_response("Не удалось получить код от MCP")
        target_variable: str = kwargs.get("target_variable", "result")
        result = await self.sandbox.execute(
            generated_code, target_variable=target_variable
        )

        # Трекинг целевых переменных

        if result.success:
            self.sandbox.last_target_variable = target_variable

            # print("last_target_variable EXECUTOR", target_variable)

            val = self.sandbox.get_variable(target_variable)

            # print("TYPE VAL", val)

            if val is not None and (hasattr(val, "shape") and hasattr(val, "head")):
                # print("last_target_variable EXECUTOR DATAFRAME", target_variable)

                self.sandbox.last_dataframe_variable = target_variable
        tool_response = self._format_response(
            result, generated_code, target_variable, **kwargs
        )
        return tool_response

    def _prepare_mcp_args(self, **kwargs: Any) -> Dict[str, Any]:
        mcp_args = {k: v for k, v in kwargs.items() if v is not None}
        mcp_args["schema_context"] = self._get_current_schema()

        # Библиотеки

        mcp_args["used_libraries"] = self._get_used_library_context()

        # Retry-контекст

        if "previous_code" not in mcp_args and self._previous_code:
            mcp_args["previous_code"] = self._previous_code
        if "error_context" not in mcp_args and self._error_context:
            mcp_args["error_context"] = self._error_context
        allowed_keys = self._get_allowed_mcp_keys()
        if allowed_keys is not None:
            filtered = {k: v for k, v in mcp_args.items() if k in allowed_keys}
            removed = set(mcp_args.keys()) - set(filtered.keys())
            if removed:
                print(
                    f"[MCP-FILTER] Убраны параметры, отсутствующие в схеме: {removed}"
                )
            mcp_args = filtered
        return mcp_args

    def _get_allowed_mcp_keys(self) -> Optional[set]:
        # Вариант 1: args_schema — Pydantic-класс

        schema = getattr(self.mcp_tool, "args_schema", None)
        if schema is not None:
            if isinstance(schema, type) and hasattr(
                schema, "model_fields"
            ):  # Pydantic v2
                return set(schema.model_fields.keys())
            if isinstance(schema, type) and hasattr(
                schema, "__fields__"
            ):  # Pydantic v1
                return set(schema.__fields__.keys())
            if isinstance(schema, dict) and "properties" in schema:
                return set(schema["properties"].keys())

        # Вариант 2: input_schema — JSON dict

        input_schema = getattr(self.mcp_tool, "input_schema", None)
        if isinstance(input_schema, dict) and "properties" in input_schema:
            return set(input_schema["properties"].keys())

        # Вариант 3: schema() метод

        if hasattr(self.mcp_tool, "schema"):
            try:
                s = self.mcp_tool.schema()
                if isinstance(s, dict) and "properties" in s:
                    return set(s["properties"].keys())
            except Exception:
                pass

        # Не удалось определить — не фильтруем

        return None

    def _get_used_library_context(self) -> str:
        if not self.sandbox.allowed_libraries:
            return "Доступны только встроенные библиотеки Python"
        allowed = self.sandbox.allowed_libraries
        return ", ".join(allowed)

    async def _invoke_mcp_and_parse(self, mcp_args: Dict[str, Any]) -> Optional[str]:
        try:
            response = await self.mcp_tool.ainvoke(mcp_args)
            return self._extract_code_from_response(response)
        except Exception:
            return None

    def _extract_code_from_response(self, raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return self._clean_code(raw_response)
        if hasattr(raw_response, "content"):
            content = raw_response.content
            if isinstance(content, str):
                return self._clean_code(content)
            if isinstance(content, list) and len(content) > 0:
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        return self._clean_code(item["text"])
                    if isinstance(item, str):
                        return self._clean_code(item)
        if isinstance(raw_response, dict):
            for key in ["code", "text", "content", "result", "output"]:
                if key in raw_response and isinstance(raw_response[key], str):
                    return self._clean_code(raw_response[key])
            if "response" in raw_response:
                return self._extract_code_from_response(raw_response["response"])
        if isinstance(raw_response, list) and len(raw_response) > 0:
            return self._extract_code_from_response(raw_response[0])
        return self._clean_code(str(raw_response))

    @staticmethod
    def _clean_code(code: str) -> str:
        code = re.sub(r"```\w*\s*\n?", "", code)
        code = re.sub(r"```\n?", "", code)
        code = re.sub(r"^\s*python\s*\n?", "", code, flags=re.IGNORECASE)
        return code.strip()

    def _format_response(
        self,
        result: ExecutionResult,
        code: str,
        target_var: str,
        **kwargs: Any,
    ) -> str:
        response: Dict[str, Any] = {
            "success": result.success,
            "tool_name": self.name,
            "generated_code": code,
            "target_variable": target_var,
            "execution_time_ms": result.execution_time_ms,
            "new_variables": result.new_variable_schemas,
        }
        if result.success:
            response["variable_preview"] = self.sandbox._get_variable_preview(
                target_var
            )
            response["message"] = f"Переменная '{target_var}' создана успешно"
        else:
            response["error"] = result.error
            response["message"] = "Выполнение кода завершилось с ошибкой"

        # Сохраняем контекст для retry

        self._previous_code = code
        self._error_context = result.error
        return json.dumps(response, ensure_ascii=False)

    @staticmethod
    def _make_error_response(message: str) -> str:
        return json.dumps(
            {"success": False, "error": message, "message": message},
            ensure_ascii=False,
        )

    def _get_current_schema(self) -> str:
        lines = []
        for name, value in self.sandbox.globals.items():
            if name.startswith("_"):
                continue
            if type(value).__name__ == "module":
                continue
            try:
                if hasattr(value, "shape") and hasattr(value, "columns"):
                    # ── Базовая информация ──

                    cols_info = ", ".join(
                        f"{col} ({dtype})" for col, dtype in value.dtypes.items()
                    )

                    # ── NaN-статистика ──

                    nan_counts = value.isna().sum()
                    total_rows = len(value)
                    nan_cols = nan_counts[nan_counts > 0]
                    if not nan_cols.empty:
                        nan_lines = []
                        for col, count in nan_cols.items():
                            pct = count / total_rows * 100
                            nan_lines.append(f"  • {col}: {count} NaN ({pct:.1f}%)")
                        nan_info = (
                            f"\n Столбцы с пропущенными значениями (NaN):\n"
                            + "\n".join(nan_lines)
                        )
                    else:
                        nan_info = "\nПропущенных значений (NaN) нет."

                    # ── Пустые строки (не NaN, но '') ──

                    str_cols = value.select_dtypes(include=["object", "string"]).columns
                    empty_str_parts = []
                    for col in str_cols:
                        empty_count = (value[col] == "").sum()
                        if empty_count > 0:
                            pct = empty_count / total_rows * 100
                            empty_str_parts.append(
                                f"  • {col}: {empty_count} пустых строк ({pct:.1f}%)"
                            )
                    if empty_str_parts:
                        empty_str_info = (
                            f"\n Столбцы с пустыми строками (''):\n"
                            + "\n".join(empty_str_parts)
                        )
                    else:
                        empty_str_info = ""
                    lines.append(
                        f"{name}: pandas DataFrame, shape={value.shape}\n"
                        f" Столбцы: {cols_info}\n"
                        f" Пример данных:\n{value.head(2).to_string()}"
                        f"{nan_info}"
                        f"{empty_str_info}"
                    )
                elif name not in self.sandbox._base_globals_names:
                    preview = str(value)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    lines.append(f"{name}: {type(value).__name__} = {preview}")
            except Exception as e:
                lines.append(f"{name}: error ({e})")
        result = "\n".join(lines)
        return result or "No variables"

    def with_context(
        self,
        previous_code: Optional[str] = None,
        error_context: Optional[str] = None,
    ) -> "BaseCodeExecutorTool":
        new_tool = self.__class__(
            mcp_tool=self.mcp_tool,
            sandbox=self.sandbox,
            name=self.name,
            description=self.description,
        )
        object.__setattr__(new_tool, "_previous_code", previous_code)
        object.__setattr__(new_tool, "_error_context", error_context)
        return new_tool

    def reset_context(self) -> None:
        self._previous_code = None
        self._error_context = None
