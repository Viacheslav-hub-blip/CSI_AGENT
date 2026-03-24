import os
import sys
import asyncio
import numpy as np
import pandas as pd
from typing import Optional
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from langchain.tools import tool
from sandbox import ClientPythonSandbox
from executor import BaseCodeExecutorTool

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

# ━━━━━━━━━━━━━━━━━━━ КОНСТАНТЫ ━━━━━━━━━━━━━━━━━━━

DATAFRAMES_DIR: str = "DataFrames"
TOOL_INVOKE_SUCCESS_FOR_AGENT: str = "Операция выполнена успешно"

# ━━━━━━━━━━━━━━━━━━━ SANDBOX ━━━━━━━━━━━━━━━━━━━

main_react_agent_sandbox = ClientPythonSandbox(
    allowed_libraries={"pd", "np", "px", "go", "pio"},
    initial_globals={
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "pio": pio,
    },
)

# ━━━━━━━━━━━━━━━ ХЕЛПЕРЫ SANDBOX ↔ DATAFRAME ━━━━━━━━━━━━━━━


def load_dataframe_to_sandbox(
    sandbox: ClientPythonSandbox,
    df: Optional[pd.DataFrame] = None,
) -> None:
    if df is not None:
        sandbox.add_variable("df", df.copy())
    else:
        path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
        if os.path.exists(path):
            current_df = pd.read_pickle(path)
            sandbox.add_variable("df", current_df)


def get_dataframe_from_sandbox(
    sandbox: ClientPythonSandbox,
) -> Optional[pd.DataFrame]:
    target = sandbox.last_dataframe_variable
    if target:
        val = sandbox.get_variable(target)
        if isinstance(val, pd.DataFrame):
            return val.copy()

    # Fallback: код мог модифицировать df напрямую

    df_val = sandbox.get_variable("df")
    if isinstance(df_val, pd.DataFrame):
        return df_val.copy()
    return None


def _safe_dataframe_preview(df: pd.DataFrame, rows: int) -> str:
    preview_df = df.head(rows)
    try:
        return preview_df.to_xml()
    except Exception:
        try:
            return preview_df.to_json()
        except Exception as exc:
            return f"Не удалось сериализовать DataFrame: {exc}"


# ━━━━━━━━━━━━━━━━━━━ MCP ИНСТРУМЕНТЫ ━━━━━━━━━━━━━━━━━━━


def get_python_code_mcp_tools() -> list:
    if MultiServerMCPClient is None:
        return []
    mcp_client = MultiServerMCPClient(
        {
            "code_generator": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8201/mcp",
            }
        }
    )
    return asyncio.run(mcp_client.get_tools())


def get_plotly_code_mcp_tools() -> list:
    if MultiServerMCPClient is None:
        return []
    mcp_client = MultiServerMCPClient(
        {
            "code_generator": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8202/mcp",
            }
        }
    )
    return asyncio.run(mcp_client.get_tools())


def _safe_load_mcp_tools(loader, label: str) -> list:
    try:
        return loader()
    except Exception as exc:
        print(f"[tools] Failed to load {label} MCP tools: {exc}")
        return []


def _make_unavailable_tool(name: str, description: str):

    def _tool(task: str = "", target_variable: str = "result") -> str:
        del task, target_variable
        return (
            f"MCP tool '{name}' is unavailable. "
            f"Start the corresponding MCP server or install missing adapters."
        )

    _tool.__name__ = name
    _tool.__doc__ = description
    return tool(name)(_tool)


python_code_mcp_tools = _safe_load_mcp_tools(get_python_code_mcp_tools, "python")
python_plotly_mcp_tools = _safe_load_mcp_tools(get_plotly_code_mcp_tools, "plotly")
mcp_generate_code_tool = next(
    (t for t in python_code_mcp_tools if t.name == "generate_python_code"),
    None,
)
mcp_generate_plotly_tool = next(
    (t for t in python_plotly_mcp_tools if t.name == "generate_plotly_python_code"),
    None,
)
if mcp_generate_code_tool is not None:
    print("mcp_generate_code_tool")
    code_tool = BaseCodeExecutorTool(
        mcp_tool=mcp_generate_code_tool,
        sandbox=main_react_agent_sandbox,
        name=mcp_generate_code_tool.name,
        description=mcp_generate_code_tool.description,
    )
else:
    code_tool = _make_unavailable_tool(
        "generate_python_code",
        "Fallback tool returned when the Python MCP code generator is unavailable.",
    )
if mcp_generate_plotly_tool is not None:
    print("mcp_generate_plotly_tool")
    graph_tool = BaseCodeExecutorTool(
        mcp_tool=mcp_generate_plotly_tool,
        sandbox=main_react_agent_sandbox,
        name=mcp_generate_plotly_tool.name,
        description=mcp_generate_plotly_tool.description,
    )
else:
    graph_tool = _make_unavailable_tool(
        "generate_plotly_python_code",
        "Fallback tool returned when the Plotly MCP code generator is unavailable.",
    )

# ━━━━━━━━━━━━━━━━━━━ ИНСТРУМЕНТЫ-ФУНКЦИИ ━━━━━━━━━━━━━━━━━━━


@tool("display_data_frame")
def display_data_frame(path_to_dataframe: str) -> str:
    """Отображает первые 5 строк DataFrame из файла.

    Поддерживает форматы: .pkl, .xlsx/.xls, .csv.
    Путь ограничивается директорией ``DATAFRAMES_DIR``.

    Args:
        path_to_dataframe: Имя файла (без директории) или полный путь.

    Returns:
        XML-представление первых 5 строк DataFrame или сообщение об ошибке.
    """
    # Ограничиваем доступ только к DATAFRAMES_DIR

    safe_path = os.path.join(DATAFRAMES_DIR, os.path.basename(path_to_dataframe))
    if not os.path.exists(safe_path):
        return f"Файл '{path_to_dataframe}' не найден в {DATAFRAMES_DIR}"
    try:
        if safe_path.endswith(".pkl"):
            df = pd.read_pickle(safe_path)
        elif safe_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(safe_path)
        elif safe_path.endswith(".csv"):
            df = pd.read_csv(safe_path)
        else:
            df = pd.read_pickle(safe_path)
        return _safe_dataframe_preview(df, rows=5)
    except Exception as e:
        return f"Не удалось загрузить файл: {e}"


@tool("calculate_base_statictics_for_column")
def calculate_base_statictics_for_column(column_name: str) -> str:
    """Вычисляет базовую статистику для числового столбца текущего DataFrame.

    Читает ``current_dataframe.pkl`` из ``DATAFRAMES_DIR``.
    Рассчитывает: count, mean, std, min, 25%, 50%, 75%, max, range,
    skewness, kurtosis, std_error.

    Args:
        column_name: Имя столбца с числовыми значениями.

    Returns:
        Строковое представление словаря со статистиками или сообщение об ошибке.
    """
    path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
    if not os.path.exists(path):
        return f"Файл {path} не найден"
    current_dataframe = pd.read_pickle(path)
    if column_name not in current_dataframe.columns:
        return (
            f"Столбец '{column_name}' отсутствует в DataFrame. "
            f"Доступные столбцы: {list(current_dataframe.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(current_dataframe[column_name]):
        return (
            f"Столбец '{column_name}' не содержит числовых значений "
            f"(тип: {current_dataframe[column_name].dtype})"
        )
    stats = current_dataframe[column_name].describe().to_dict()
    stats["range"] = stats["max"] - stats["min"]
    stats["skewness"] = float(current_dataframe[column_name].skew())
    stats["kurtosis"] = float(current_dataframe[column_name].kurtosis())
    stats["std_error"] = float(current_dataframe[column_name].sem())
    return str(stats)


@tool("get_available_dataframes")
def get_available_dataframes() -> str:
    """Возвращает список доступных DataFrame в директории ``DATAFRAMES_DIR``.

    Returns:
        Многострочная строка с именами .pkl файлов или сообщение об ошибке / пустом списке.
    """
    if not os.path.exists(DATAFRAMES_DIR):
        return f"Директория {DATAFRAMES_DIR} не найдена"
    files = os.listdir(DATAFRAMES_DIR)
    pkl_files = [f for f in files if f.endswith(".pkl")]
    if not pkl_files:
        return "Нет доступных DataFrame"
    return "\n".join(pkl_files)


class ChangeCurrentDataframeParams(BaseModel):
    new_dataframe_name: str = Field(
        description=(
            "Название DataFrame, который должен стать текущим, "
            "в формате: 'название.pkl'"
        ),
    )


@tool("change_current_dataframe", args_schema=ChangeCurrentDataframeParams)
def change_current_dataframe(new_dataframe_name: str) -> str:
    """Заменяет текущий DataFrame на указанный.

    Читает файл из ``DATAFRAMES_DIR`` и перезаписывает ``current_dataframe.pkl``.

    Args:
        new_dataframe_name: Имя .pkl файла из DATAFRAMES_DIR.

    Returns:
        Сообщение об успехе или ошибке.
    """
    new_path = os.path.join(DATAFRAMES_DIR, new_dataframe_name)
    if not os.path.exists(new_path):
        return f"Файл '{new_dataframe_name}' не найден в {DATAFRAMES_DIR}"
    new_df = pd.read_pickle(new_path)
    current_path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
    new_df.to_pickle(current_path)
    return TOOL_INVOKE_SUCCESS_FOR_AGENT


@tool("show_current_uses_dataframe")
def show_current_uses_dataframe() -> str:
    """Отображает структуру текущего DataFrame.

    Читает ``current_dataframe.pkl`` из ``DATAFRAMES_DIR`` и возвращает XML первых 2 строк + типы данных.

    Returns:
        Строка с информацией о DataFrame или сообщение об ошибке.
    """
    path = os.path.join(DATAFRAMES_DIR, "current_dataframe.pkl")
    if not os.path.exists(path):
        return "Текущий DataFrame не найден"
    current_df = pd.read_pickle(path)
    dataframe_xml = _safe_dataframe_preview(current_df, rows=2)
    return (
        f"Текущий DataFrame ({current_df.shape[0]} строк, "
        f"{current_df.shape[1]} столбцов):\n"
        f"{dataframe_xml}\n\n"
    )
