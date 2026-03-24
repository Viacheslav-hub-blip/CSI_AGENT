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
STAT_AGENT_TOOLS = [calculate_base_statictics_for_column]
PURPOSE_AGENT_TOOLS = [
    get_available_dataframes,
    change_current_dataframe,
    show_current_uses_dataframe,
]


def get_react_main_agent(prompt: str):
    return create_react_agent(llm, tools=MAIN_AGENT_TOOLS, prompt=prompt)


def get_react_stat_agent(prompt: str):
    return create_react_agent(llm, tools=STAT_AGENT_TOOLS, prompt=prompt)


def get_react_purpose_agent(prompt: str):
    return create_react_agent(llm, tools=PURPOSE_AGENT_TOOLS, prompt=prompt)
