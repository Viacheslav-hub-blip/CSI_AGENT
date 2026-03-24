import ast
import asyncio
import contextlib
import io
import traceback
import time
from typing import Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: Optional[str] = None
    new_variable_schemas: Dict[str, str] = field(default_factory=dict)
    execution_time_ms: int = 0


class CodeValidator:
    DANGEROUS_CALLS: Set[str] = {"eval", "exec", "compile", "__import__"}
    DANGEROUS_MODULES: Set[str] = {"os", "sys", "subprocess", "socket", "shutil"}

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in cls.DANGEROUS_CALLS
                ):
                    return False, f"Dangerous function call: {node.func.id}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in cls.DANGEROUS_MODULES:
                        return False, f"Dangerous module import: {alias.name}"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module in cls.DANGEROUS_MODULES:
                    return False, f"Dangerous module import: {node.module}"
        return True, ""


class ClientPythonSandbox:
    def __init__(
        self,
        allowed_libraries: Optional[Set[str]] = None,
        initial_globals: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.allowed_libraries: Set[str] = allowed_libraries or set()
        self.globals: Dict[str, Any] = initial_globals.copy() if initial_globals else {}
        self._base_globals_names: Set[str] = set(self.globals.keys())
        self._lock: asyncio.Lock = asyncio.Lock()
        self.last_target_variable: Optional[str] = None
        self.last_dataframe_variable: Optional[str] = None

    async def execute(
        self, code: str, target_variable: Optional[str] = None
    ) -> ExecutionResult:
        start_time = time.time()
        is_valid, error_msg = CodeValidator.validate(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=f"Code validation failed: {error_msg}",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        async with self._lock:
            keys_before = set(self.globals.keys())
            try:
                with (
                    contextlib.redirect_stdout(stdout_capture),
                    contextlib.redirect_stderr(stderr_capture),
                ):
                    exec(code, self.globals, self.globals)
                success = True
                error_msg = None
            except Exception:
                success = False
                error_msg = traceback.format_exc()
            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            keys_after = set(self.globals.keys())
            new_keys = keys_after - keys_before
            new_schemas: Dict[str, str] = {}
            for key in new_keys:
                if not key.startswith("__"):
                    new_schemas[key] = self._get_variable_preview(key)

            # Проверяем наличие target_variable
            if target_variable and target_variable not in self.globals:
                if not new_keys:
                    success = False
                    error_msg = (
                        f"Target variable '{target_variable}' not found "
                        f"after execution and no new variables were created."
                    )
            return ExecutionResult(
                success=success,
                output=output,
                error=error_msg,
                new_variable_schemas=new_schemas,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    def _get_variable_preview(self, var_name: str) -> str:
        if var_name not in self.globals:
            return "Variable not found"
        val = self.globals[var_name]
        try:
            if hasattr(val, "shape") and hasattr(val, "head"):
                return f"DataFrame Shape: {val.shape}\n{val.head(2).to_string()}"
            elif hasattr(val, "to_string"):
                return val.to_string()
            else:
                s_val = str(val)
                return s_val[:500] + "..." if len(s_val) > 500 else s_val
        except Exception as e:
            return f"Error getting preview: {e}"

    def get_all_variable_previews(self) -> Dict[str, str]:
        schemas: Dict[str, str] = {}
        for name, value in self.globals.items():
            if name.startswith("_") or name in self._base_globals_names:
                continue
            if type(value).__name__ == "module":
                continue
            schemas[name] = self._get_variable_preview(name)
        return schemas

    def get_variable(self, var_name: str) -> Any:
        return self.globals.get(var_name)

    def add_variable(
        self, name: str, value: Any, exclude_from_preview: bool = False
    ) -> None:
        self.globals[name] = value
        if exclude_from_preview:
            self._base_globals_names.add(name)

    def reset(self, keep_base: bool = True) -> None:
        if keep_base:
            current_base = {
                k: v for k, v in self.globals.items() if k in self._base_globals_names
            }
            self.globals = current_base
        else:
            self.globals = {}
        self._base_globals_names = set()
        self.last_target_variable = None
        self.last_dataframe_variable = None
