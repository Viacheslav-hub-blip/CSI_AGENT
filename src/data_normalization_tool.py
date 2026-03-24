import pandas as pd
import numpy as np
import uuid
import json
import warnings
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.config.model import model as llm
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
np.random.seed(42)
n_rows = 150

# df = pd.DataFrame({

#     "col_1_id": [str(uuid.uuid4())[:8] for _ in range(n_rows)],

#     "counterparty_raw": np.random.choice(

#         ["Сбербанк", "ПАО Сбер", "сбербанк", "Сбер", "Газпром", "ООО Газпром", "Т-Банк", "Tinkoff", "Тиньк", "Тинек",

#          "Альфа банк", "Альфа", "Альфа-банк"],

#         n_rows),

#     "time_created": np.random.choice(["12.05.2023", "12/05/23", "12 мая 2023", "May 12th, 2023"], n_rows),

#     "money_value": np.random.uniform(1000, 50000, n_rows).round(2)

# })

df = pd.read_csv("src/dirty_dataset.csv")

# ==========================================

# 1. СХЕМЫ ДАННЫХ

# ==========================================


class ColumnPlan(BaseModel):
    original_name: str = Field(description="Исходное название")
    new_name: str = Field(description="Новое чистое название")
    action_type: str = Field(
        description="СТРОГО ОДНО ИЗ ТРЕХ: 'CLUSTER' (группировка компаний/текста), 'FORMAT' (форматирование дат/чисел) или 'PASS' (пропустить, не трогать)"
    )
    instruction: str = Field(
        description="Инструкция. Например: 'Сгруппируй бренды' или 'Приведи к YYYY-MM-DD'"
    )


class DatasetPlan(BaseModel):
    columns: List[ColumnPlan]


class Cluster(BaseModel):
    golden_key: str
    variations: List[str]


class ClusterOutput(BaseModel):
    clusters: List[Cluster]


class FormatMapping(BaseModel):
    original: str
    formatted: str


class FormatOutput(BaseModel):
    mappings: List[FormatMapping]


class ColumnTransformConfig(BaseModel):
    col_name: str
    new_name: str
    type: str
    clusters: List[Cluster] = []
    formats: List[FormatMapping] = []


class AllTransforms(BaseModel):
    columns: List[ColumnTransformConfig]


class AgentState(TypedDict):
    raw_records: List[Dict[str, Any]]
    sample_md: str
    plan: List[Dict[str, Any]]
    transforms: List[Dict[str, Any]]
    manual_transforms: List[Dict[str, Any]]
    final_df: Any


class UserIntent(BaseModel):
    action: Literal["PROCEED", "EDIT", "UNDO", "UNKNOWN"] = Field(
        description="""Определи намерение пользователя:
        - PROCEED: Пользователь согласен с предложенным планом и хочет продолжить/применить. (например: "ок", "всё верно", "погнали", "согласен", "давай")
        - EDIT: Пользователь хочет изменить текущие правила маппинга. (например: "замени х на у", "объедини эти два", "не трогай дату")
        - UNDO: Пользователь хочет отменить последнее действие или вернуться на шаг назад. (например: "откати", "верни как было", "отмена", "назад")
        - UNKNOWN: Обычный разговор, вопрос или неясная команда."""
    )
    extracted_instruction: str = Field(
        description="Если action == 'EDIT', вытащи суть правки в повелительном наклонении. Иначе верни пустую строку."
    )
    agent_reply: str = Field(
        description="Если action == 'UNKNOWN', напиши вежливый ответ, объясняющий, что ты сейчас ждешь подтверждения или правок таблиц."
    )


# ==========================================

# 2. ИНИЦИАЛИЗАЦИЯ LLM И ПРОМПТОВ

# ==========================================

profiler_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты Senior Data Architect. Определи тип преобразования для каждой колонки: 'PASS', 'CLUSTER' или 'FORMAT'.
        КРИТИЧЕСКОЕ ПРАВИЛО ДЛЯ ПОЛЯ 'instruction':
        Пиши ТОЛЬКО техническое правило БЕЗ примеров из данных.
        ✅ Правильно: 'Сгруппировать названия по родительскому бренду'.
        ❌ Неправильно: 'Объедини Сбер и Газпром'.
        """,
            ),
            ("user", "Пример данных:\n{sample_data}"),
        ]
    )
    | RunnablePassthrough(lambda x: print(x))
    | llm.with_structured_output(DatasetPlan)
)
cluster_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты эксперт по очистке данных. Твоя задача — разбить входной список значений на логические группы.
        Инструкция от архитектора: {instruction}
        ЖЕСТКИЕ ПРАВИЛА:
        1. Каждое значение из списка `uniques` должно попасть ровно в один кластер.
        2. В массив `variations` КОПИРУЙ строки строго 1 в 1 из входного списка.
        3. ЗАПРЕЩЕНО изменять входные строки, придумывать новые слова или склеивать разные названия (Сбербанк и Газпром — это два РАЗНЫХ ключа).
        """,
            ),
            ("user", "Входной список уникальных значений:\n{uniques}"),
        ]
    )
    | RunnablePassthrough(lambda x: print(x))
    | llm.with_structured_output(ClusterOutput)
)
format_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Отформатируй каждое значение по правилу. Инструкция: {instruction}",
            ),
            ("user", "Уникальные значения:\n{uniques}"),
        ]
    )
    | RunnablePassthrough(lambda x: print(x))
    | llm.with_structured_output(FormatOutput)
)
update_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты редактор правил очистки данных. Тебе передан JSON с правилами и правка пользователя. Измени JSON согласно правке. Верни все колонки.",
            ),
            ("user", "Текущие правила:\n{current_state}\n\nКоманда: {instruction}"),
        ]
    )
    | RunnablePassthrough(lambda x: print(x))
    | llm.with_structured_output(AllTransforms)
)

# --- НОВЫЙ РОУТЕР ДЛЯ ПОНИМАНИЯ НАМЕРЕНИЙ ПОЛЬЗОВАТЕЛЯ ---

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты AI-ассистент, управляющий графом нормализации данных.
    Проанализируй ответ пользователя и выбери правильный action.""",
        ),
        ("user", "Ответ пользователя: {user_input}"),
    ]
)
intent_router = (
    router_prompt
    | RunnablePassthrough(lambda x: print(x))
    | llm.with_structured_output(UserIntent)
)

# ==========================================

# 3. УЗЛЫ ГРАФА

# ==========================================


def profile_dataset(state: AgentState) -> Dict[str, Any]:
    plan = profiler_chain.invoke({"sample_data": state["sample_md"]})
    return {"plan": [c.model_dump() for c in plan.columns]}


def build_transformations(state: AgentState) -> Dict[str, Any]:
    temp_df = pd.DataFrame(state["raw_records"])
    transforms = []
    for col in state["plan"]:
        orig_name = col["original_name"]
        if col["action_type"] == "PASS":
            continue
        uniques = temp_df[orig_name].dropna().astype(str).unique().tolist()
        uniqueness_ratio = len(uniques) / len(temp_df[orig_name].dropna())
        if uniqueness_ratio >= 0.9 and col["action_type"] == "CLUSTER":
            print(
                f"🛡️ [Система]: Отменена кластеризация для '{orig_name}' (выглядит как колонка уникальных ID)."
            )
            continue
        if col["action_type"] == "CLUSTER":
            res = cluster_chain.invoke(
                {"uniques": uniques, "instruction": col["instruction"]}
            )
            transforms.append(
                {
                    "col_name": orig_name,
                    "new_name": col["new_name"],
                    "type": "CLUSTER",
                    "clusters": [c.model_dump() for c in res.clusters],
                    "formats": [],
                }
            )
        elif col["action_type"] == "FORMAT":
            res = format_chain.invoke(
                {"uniques": uniques, "instruction": col["instruction"]}
            )
            transforms.append(
                {
                    "col_name": orig_name,
                    "new_name": col["new_name"],
                    "type": "FORMAT",
                    "clusters": [],
                    "formats": [f.model_dump() for f in res.mappings],
                }
            )
    return {"transforms": transforms}


def apply_transformations(state: AgentState) -> Dict[str, Any]:
    final_df = pd.DataFrame(state["raw_records"])
    active_transforms = state.get("manual_transforms") or state["transforms"]
    for t in active_transforms:
        old_col = t["col_name"]
        new_col = t["new_name"]
        mapping_dict = {}
        if t["type"] == "CLUSTER":
            for c in t["clusters"]:
                for v in c["variations"]:
                    mapping_dict[v] = c["golden_key"]
        elif t["type"] == "FORMAT":
            for f in t["formats"]:
                mapping_dict[f["original"]] = f["formatted"]
        final_df[new_col] = (
            final_df[old_col].astype(str).map(mapping_dict).fillna(final_df[old_col])
        )
    return {"final_df": final_df.to_dict(orient="records")}


builder = StateGraph(AgentState)
builder.add_node("profile_dataset", profile_dataset)
builder.add_node("build_transformations", build_transformations)
builder.add_node("apply_transformations", apply_transformations)
builder.add_edge(START, "profile_dataset")
builder.add_edge("profile_dataset", "build_transformations")
builder.add_edge("build_transformations", "apply_transformations")
builder.add_edge("apply_transformations", END)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["apply_transformations"])

# ==========================================

# 4. ТЕСТИРОВАНИЕ И КОНСОЛЬНЫЙ ЦИКЛ С РОУТЕРОМ

# ==========================================

if __name__ == "__main__":
    sample_md = df.head(5).to_markdown(index=False)
    raw_records = df.to_dict(orient="records")
    initial_state = {
        "raw_records": raw_records,
        "sample_md": sample_md,
        "plan": [],
        "transforms": [],
        "manual_transforms": [],
    }
    thread_config = {"configurable": {"thread_id": "smart_agent_session_1"}}
    print(
        "🤖 Агент: Изучаю колонки и готовлю словари преобразований (это займет пару секунд)..."
    )
    for event in graph.stream(initial_state, thread_config):
        pass
    while True:
        state = graph.get_state(thread_config)
        if not state.next:
            break
        active_transforms = state.values.get("manual_transforms") or state.values.get(
            "transforms", []
        )
        print("\n" + "=" * 60)
        print("🤖 Агент: ⚠️ ПРЕДЛАГАЮ СЛЕДУЮЩИЕ ПРЕОБРАЗОВАНИЯ ДЛЯ БАЗЫ:")
        print("=" * 60 + "\n")
        for t in active_transforms:
            print(f"🛠️ Колонка [{t['col_name']}] -> [{t['new_name']}] ({t['type']}):")
            if t["type"] == "CLUSTER":
                for c in t["clusters"]:
                    vars_str = ", ".join(c["variations"])
                    print(f"  🔹 {c['golden_key']:<15} <--- [{vars_str}]")
            elif t["type"] == "FORMAT":
                for f in t["formats"]:
                    print(f"  🔸 '{f['original']}' -> '{f['formatted']}'")
            print()
        print("-" * 60)

        # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Интерактивный диалог вместо жестких команд ---

        print(
            "🤖 Агент: Жду ваших указаний. Применяем план, или нужно что-то исправить?"
        )
        user_input = input("🧑‍💻 Вы: ").strip()
        if not user_input:
            continue
        print("🤖 Агент: Анализирую ваше сообщение...")
        intent = intent_router.invoke({"user_input": user_input})
        if intent.action == "PROCEED":
            print("🤖 Агент: Отлично! Применяю изменения ко всем строкам таблицы...")
            for event in graph.stream(None, thread_config):
                pass
        elif intent.action == "EDIT":
            print(f"🤖 Агент: Вношу правки: '{intent.extracted_instruction}'...")
            current_json = json.dumps(active_transforms, ensure_ascii=False)
            updated_res = update_chain.invoke(
                {
                    "current_state": current_json,
                    "instruction": intent.extracted_instruction,
                }
            )
            updated_dicts = [c.model_dump() for c in updated_res.columns]
            graph.update_state(
                state.config,
                {"manual_transforms": updated_dicts},
                as_node="build_transformations",
            )
            print("🤖 Агент: ✨ Словари успешно обновлены!")
        elif intent.action == "UNDO":
            print("🤖 Агент: Выполняю откат к предыдущему состоянию...")
            history = list(graph.get_state_history(thread_config))
            if len(history) > 1:
                thread_config["configurable"]["checkpoint_id"] = history[1].config[
                    "configurable"
                ]["checkpoint_id"]
                print("🤖 Агент: ⏪ Состояние восстановлено.")
            else:
                print("🤖 Агент: ❌ Дальше откатывать некуда.")
        elif intent.action == "UNKNOWN":
            print(f"🤖 Агент: {intent.agent_reply}")

    # Финальный вывод

    final_state = graph.get_state(thread_config)
    if "final_df" in final_state.values:
        result_df = pd.DataFrame(final_state.values["final_df"])
        print("\n✨ --- ИТОГОВАЯ ТАБЛИЦА (Добавлены новые очищенные колонки) --- ✨")
        columns_to_show = [
            "col_1_id",
            "counterparty_raw",
            "counterparty",
            "time_created",
            "created_at",
        ]
        existing_cols = [col for col in columns_to_show if col in result_df.columns]

        # Выводим 10 случайных строк, чтобы было видно разнообразие

        print(
            result_df[existing_cols]
            .sample(10, random_state=42)
            .to_markdown(index=False)
        )
