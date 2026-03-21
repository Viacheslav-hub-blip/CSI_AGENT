"""
Фабрики суб-агентов (ReAct).
Каждая функция создаёт create_react_agent с нужным набором инструментов.
Промпт передаётся как строка (system message).
"""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langgraph.prebuilt import create_react_agent
from model import GigaChat_Max as llm
from tools import (
    calculate_base_statictics_for_column,
    get_available_dataframes,
    display_data_frame,
    change_current_dataframe,
    show_current_uses_dataframe,
    code_tool,
    graph_tool,
)

MAIN_AGENT_TOOLS = [code_tool, graph_tool, display_data_frame]
"""Инструменты для агента генерации кода и визуализации."""

STAT_AGENT_TOOLS = [calculate_base_statictics_for_column]
"""Инструменты для агента вычисления статистики."""

PURPOSE_AGENT_TOOLS = [
    get_available_dataframes,
    change_current_dataframe,
    show_current_uses_dataframe,
]
"""Инструменты для агента общего назначения."""


def get_react_main_agent(prompt: str):
    """Создаёт ReAct-агент для генерации кода и построения графиков.

    Args:
        prompt: Системный промпт (строка). create_react_agent использует его как system message.

    Returns:
        CompiledGraph — скомпилированный граф агента.
    """
    return create_react_agent(llm, tools=MAIN_AGENT_TOOLS, prompt=prompt)


def get_react_stat_agent(prompt: str):
    """Создаёт ReAct-агент для вычисления статистики по столбцам.

    Args:
        prompt: Системный промпт (строка).

    Returns:
        CompiledGraph — скомпилированный граф агента.
    """
    return create_react_agent(llm, tools=STAT_AGENT_TOOLS, prompt=prompt)


def get_react_purpose_agent(prompt: str):
    """Создаёт ReAct-агент общего назначения.

    Args:
        prompt: Системный промпт (строка).

    Returns:
        CompiledGraph — скомпилированный граф агента.
    """
    return create_react_agent(llm, tools=PURPOSE_AGENT_TOOLS, prompt=prompt)