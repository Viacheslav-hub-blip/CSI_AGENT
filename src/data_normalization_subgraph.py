import json
import threading
import warnings
from typing import Any, Literal
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
from config.model import model as llm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
DEFAULT_SAMPLE_ROWS = 5
PREVIEW_ITEMS_PER_COLUMN = 8
FORMAT_BATCH_MAX_VALUES = 100
FORMAT_BATCH_MAX_CHARS = 6000
PLAN_ACTION_TYPES = {"CLUSTER", "FORMAT", "PASS"}
TRANSFORM_ACTION_TYPES = {"CLUSTER", "FORMAT"}


def _first_list_payload(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, list):
                return item
    return None


def _first_matching_literal(
    payload: dict[str, Any],
    *,
    allowed_values: set[str],
    exclude_keys: set[str] | None = None,
) -> str | None:
    excluded = exclude_keys or set()
    for key, item in payload.items():
        if key in excluded or not isinstance(item, str):
            continue
        candidate = item.strip().upper()
        if candidate in allowed_values:
            return candidate
    return None


def _first_remaining_text(
    payload: dict[str, Any],
    *,
    exclude_keys: set[str] | None = None,
) -> str | None:
    excluded = exclude_keys or set()
    for key, item in payload.items():
        if key in excluded or not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate:
            return candidate
    return None


def _extract_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _debug_dump(value: Any) -> str:
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump(), ensure_ascii=False, indent=2, default=str)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _invoke_with_debug(
    *,
    label: str,
    chain: Any,
    debug_chain: Any,
    payload: dict[str, Any],
) -> Any:
    print(f"[normalization][{label}] payload:\n{_debug_dump(payload)}")
    try:
        result = chain.invoke(payload)
        print(f"[normalization][{label}] parsed result:\n{_debug_dump(result)}")
        return result
    except Exception as exc:
        print(f"[normalization][{label}] structured invoke failed: {exc}")
        try:
            raw_response = debug_chain.invoke(payload)
            print(
                f"[normalization][{label}] raw model output:\n{_extract_response_text(raw_response)}"
            )
        except Exception as debug_exc:
            print(f"[normalization][{label}] raw debug invoke failed: {debug_exc}")
        raise


class ColumnPlan(BaseModel):
    original_name: str = Field(description="Original dataframe column name")
    new_name: str = Field(description="Target column name after normalization")
    action_type: Literal["CLUSTER", "FORMAT", "PASS"] = Field(
        description="Normalization action for the column"
    )
    instruction: str = Field(
        description="Generic technical normalization instruction for the column"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_column_plan_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault(
            "original_name",
            payload.get("column")
            or payload.get("column_name")
            or payload.get("col_name")
            or payload.get("name"),
        )
        payload.setdefault(
            "new_name",
            payload.get("normalized_name")
            or payload.get("target_name")
            or payload.get("new_column_name")
            or payload.get("original_name")
            or payload.get("column"),
        )
        payload.setdefault(
            "action_type",
            payload.get("action_type")
            or payload.get("action")
            or payload.get("type")
            or _first_matching_literal(
                payload,
                allowed_values=PLAN_ACTION_TYPES,
                exclude_keys={"original_name", "new_name"},
            ),
        )
        payload.setdefault(
            "instruction",
            payload.get("instruction")
            or payload.get("description")
            or payload.get("details")
            or payload.get("rule")
            or _first_remaining_text(
                payload,
                exclude_keys={
                    "original_name",
                    "new_name",
                    "action_type",
                    "action",
                    "type",
                    "column",
                },
            ),
        )
        if isinstance(payload.get("action_type"), str):
            payload["action_type"] = payload["action_type"].strip().upper()
        return payload


class DatasetPlan(BaseModel):
    columns: list[ColumnPlan]

    @model_validator(mode="before")
    @classmethod
    def _coerce_columns_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "columns" in value:
            return value
        candidate = _first_list_payload(value)
        if candidate is not None:
            return {"columns": candidate}
        return value


class Cluster(BaseModel):
    golden_key: str
    variations: list[str]

    @model_validator(mode="before")
    @classmethod
    def _coerce_cluster_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault(
            "golden_key",
            payload.get("canonical")
            or payload.get("canonical_value")
            or payload.get("canonical_name")
            or payload.get("key"),
        )
        payload.setdefault(
            "variations",
            payload.get("variations") or payload.get("values") or payload.get("items"),
        )
        return payload


class ClusterOutput(BaseModel):
    clusters: list[Cluster]

    @model_validator(mode="before")
    @classmethod
    def _coerce_clusters_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "clusters" in value:
            return value
        candidate = _first_list_payload(value)
        if candidate is not None:
            return {"clusters": candidate}
        return value


class FormatMapping(BaseModel):
    original: str
    formatted: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_format_mapping_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault(
            "original",
            payload.get("source") or payload.get("value") or payload.get("input"),
        )
        payload.setdefault(
            "formatted",
            payload.get("normalized") or payload.get("target") or payload.get("output"),
        )
        return payload


class FormatOutput(BaseModel):
    mappings: list[FormatMapping]

    @model_validator(mode="before")
    @classmethod
    def _coerce_mappings_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "mappings" in value:
            return value
        candidate = _first_list_payload(value)
        if candidate is not None:
            return {"mappings": candidate}
        return value


class ColumnTransformConfig(BaseModel):
    col_name: str
    new_name: str
    type: Literal["CLUSTER", "FORMAT"]
    instruction: str = ""
    clusters: list[Cluster] = Field(default_factory=list)
    formats: list[FormatMapping] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_transform_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault(
            "col_name",
            payload.get("column")
            or payload.get("column_name")
            or payload.get("original_name")
            or payload.get("name"),
        )
        payload.setdefault(
            "new_name",
            payload.get("normalized_name")
            or payload.get("target_name")
            or payload.get("new_column_name")
            or payload.get("col_name")
            or payload.get("column"),
        )
        payload.setdefault(
            "type",
            payload.get("type")
            or payload.get("action_type")
            or payload.get("action")
            or _first_matching_literal(
                payload,
                allowed_values=TRANSFORM_ACTION_TYPES,
                exclude_keys={"col_name", "new_name"},
            ),
        )
        payload.setdefault(
            "instruction",
            payload.get("instruction")
            or payload.get("rule")
            or payload.get("description")
            or payload.get("details")
            or "",
        )
        payload.setdefault(
            "clusters", payload.get("groups") or payload.get("clusterings") or []
        )
        payload.setdefault(
            "formats", payload.get("mappings") or payload.get("formatting") or []
        )
        if isinstance(payload.get("type"), str):
            payload["type"] = payload["type"].strip().upper()
        return payload


class AllTransforms(BaseModel):
    columns: list[ColumnTransformConfig]

    @model_validator(mode="before")
    @classmethod
    def _coerce_transforms_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "columns" in value:
            return value
        candidate = _first_list_payload(value)
        if candidate is not None:
            return {"columns": candidate}
        return value


class NormalizationIntent(BaseModel):
    action: str | None = Field(
        default=None,
        description=(
            "Classify the user reply during an active normalization review. "
            "PROCEED means apply the current draft. EDIT means change the draft. "
            "UNDO means revert the latest draft edit. CANCEL means stop the whole "
            "normalization session. UNKNOWN means anything else."
        ),
    )
    extracted_instruction: str = Field(
        default="",
        description=(
            "If action is EDIT, rewrite the user correction as a short imperative "
            "instruction. Otherwise return an empty string."
        ),
    )
    agent_reply: str = Field(
        default="",
        description=(
            "If action is UNKNOWN, explain in natural language that the agent is "
            "waiting for confirmation, a correction, undo of the latest draft "
            "change, or cancellation of the session."
        ),
    )


class NormalizationIntentResolution(BaseModel):
    action: Literal["PROCEED", "EDIT", "UNDO", "CANCEL", "UNKNOWN"] = Field(
        description="Canonical intent label for the active normalization session."
    )
    extracted_instruction: str = Field(
        default="",
        description=(
            "If action is EDIT, rewrite the user correction as a short imperative "
            "instruction. Otherwise return an empty string."
        ),
    )
    agent_reply: str = Field(
        default="",
        description=(
            "If action is UNKNOWN, explain in natural language that the agent is "
            "waiting for confirmation, a correction, undo of the latest draft "
            "change, or cancellation of the session."
        ),
    )


class NormalizationRequestDecision(BaseModel):
    is_normalization_request: bool | None = Field(
        default=None,
        description=(
            "True if the user is asking to normalize, standardize, unify, clean, "
            "or canonicalize the current dataframe values."
        ),
    )
    action: str | None = Field(
        default=None,
        description=(
            "Optional action-style label if the model emits an auxiliary semantic "
            "decision instead of filling the boolean field directly."
        ),
    )
    rationale: str = Field(
        default="",
        description="Optional explanation for the decision.",
    )


class NormalizationRequestResolution(BaseModel):
    is_normalization_request: bool = Field(
        description="Final boolean decision about whether to start normalization."
    )


class NormalizationScopeDecision(BaseModel):
    target_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Exact dataframe column names explicitly or semantically targeted by the user. "
            "Return an empty list when the request applies to the dataframe broadly."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_scope_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "target_columns" in value:
            return value
        if isinstance(value, dict):
            candidate = (
                value.get("columns")
                or value.get("selected_columns")
                or value.get("column_names")
            )
            if isinstance(candidate, list):
                return {"target_columns": candidate}
        candidate = _first_list_payload(value)
        if candidate is not None:
            return {"target_columns": candidate}
        return value


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    unique_count: int
    uniqueness_ratio: float
    sample_values: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    rows: int
    columns: list[ColumnProfile]


class NormalizationGraphState(TypedDict):
    raw_records: list[dict[str, Any]]
    sample_md: str
    target_columns: list[str]
    plan: list[dict[str, Any]]
    transforms: list[dict[str, Any]]
    manual_transforms: list[dict[str, Any]]
    final_df: Any


class NormalizationOutcome(BaseModel):
    status: Literal[
        "preview",
        "preview_updated",
        "committed",
        "cancelled",
        "missing_session",
    ]
    message: str
    dataframe: Any | None = None


profiler_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You design normalization plans for dataframes.\n"
                "The runtime will collect your answer into the schema field `columns`.\n"
                "Create exactly one plan item for every dataframe column from the dataset profile.\n"
                "For each item fill these fields semantically and explicitly:\n"
                "- `original_name`: copy the exact dataframe column name.\n"
                "- `new_name`: provide a clean target column name; keep the original name when no rename is needed.\n"
                "- `action_type`: choose exactly one of CLUSTER, FORMAT, PASS.\n"
                "- `instruction`: write a short technical normalization instruction, not a label.\n"
                "Action semantics:\n"
                "- PASS: leave the column unchanged.\n"
                "- CLUSTER: merge semantically equivalent labels into canonical values.\n"
                "- FORMAT: rewrite values into one consistent representation without changing meaning.\n"
                "Critical instruction rules:\n"
                "- Never put only PASS, CLUSTER, or FORMAT into `instruction`.\n"
                "- `instruction` must be a concrete imperative sentence that explains how to normalize the column.\n"
                "- For PASS use an instruction such as leaving the column unchanged.\n"
                "- For CLUSTER and FORMAT, keep the instruction generic and technical.\n"
                "- Do not mention concrete sample values, quoted literals, or specific example mappings from the dataset.\n"
                "- Do not instruct the downstream step to output null, None, NaN, or empty canonical values.\n"
                "- If missing-like values should be normalized, describe that semantically as grouping placeholders under one descriptive canonical label.\n"
                "- Base the decision on the dataset profile and preview, not on brittle lexical shortcuts."
            ),
        ),
        (
            "user",
            "Dataset profile:\n{dataset_profile}\n\nSample preview:\n{sample_data}",
        ),
    ]
)
profiler_chain = profiler_prompt | llm.with_structured_output(DatasetPlan)
profiler_debug_chain = profiler_prompt | llm
cluster_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You normalize one categorical dataframe column.\n"
                "The runtime expects the top-level schema field `clusters`.\n"
                "Return only cluster items for that field. Do not wrap the result into an extra object keyed by the column name.\n"
                "Column name: {column_name}\n"
                "Column profile:\n{column_profile}\n"
                "Architect instruction: {instruction}\n"
                "Each cluster item must contain:\n"
                "- `golden_key`: the canonical normalized value for the group.\n"
                "- `variations`: the list of original source values that belong to that group.\n"
                "Rules:\n"
                "1. Every input value from `uniques` must appear in exactly one cluster.\n"
                "2. Copy every source value into `variations` exactly as-is.\n"
                "3. Do not invent new source variations.\n"
                "4. Do not merge clearly different real-world entities.\n"
                "5. `golden_key` must always be a non-empty string.\n"
                "6. Never use null, None, NaN, or an empty string for `golden_key`.\n"
                "7. If a group represents placeholders, missing values, empty strings, or unknown markers, choose a descriptive string canonical label for that group instead of null.\n"
                "8. Do not return dictionaries of dictionaries, mappings keyed by value, or a top-level key named after the column.\n"
                "9. If the architect instruction seems to imply null-like output, preserve the intent but convert it into a descriptive non-null string label because the schema requires a string key."
            ),
        ),
        ("user", "Unique values:\n{uniques}"),
    ]
)
cluster_chain = cluster_prompt | llm.with_structured_output(ClusterOutput)
cluster_debug_chain = cluster_prompt | llm
format_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You normalize one dataframe column by formatting a bounded batch of source values.\n"
                "The runtime expects the top-level schema field `mappings`.\n"
                "Return only mapping items for that field.\n"
                "Column name: {column_name}\n"
                "Column profile:\n{column_profile}\n"
                "Instruction: {instruction}\n"
                "Each mapping item must contain:\n"
                "- `original`: the exact source value.\n"
                "- `formatted`: the normalized representation.\n"
                "Rules:\n"
                "1. Return one mapping for every source value in this batch.\n"
                "2. Preserve the original meaning of each value.\n"
                "3. Do not wrap the answer into an extra object keyed by the column name.\n"
                "4. Do not omit `original` or `formatted`.\n"
                "5. Follow the instruction as a reusable normalization rule, even though you only see one batch of values."
            ),
        ),
        ("user", "Unique values:\n{uniques}"),
    ]
)
format_chain = format_prompt | llm.with_structured_output(FormatOutput)
format_debug_chain = format_prompt | llm
update_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You edit a normalization draft according to a user correction.\n"
                "The runtime expects the full updated transformation list in the top-level schema field `columns`.\n"
                "Return the full updated list of active transformations.\n"
                "Keep unaffected transformations unchanged.\n"
                "If the user wants a column left untouched, remove that column from the active transformation list.\n"
                "For each transformation item fill these exact fields:\n"
                "- `col_name`\n"
                "- `new_name`\n"
                "- `type` with exactly CLUSTER or FORMAT\n"
                "- `instruction`\n"
                "- `clusters`\n"
                "- `formats`\n"
                "For rule-driven FORMAT transforms, keep `formats` empty unless the user explicitly asks for concrete value-level mappings.\n"
                "Do not wrap the result into any extra keys except the schema field."
            ),
        ),
        (
            "user",
            (
                "Current active transformations:\n{current_state}\n\n"
                "User correction:\n{instruction}"
            ),
        ),
    ]
)
update_chain = update_prompt | llm.with_structured_output(AllTransforms)
update_debug_chain = update_prompt | llm
intent_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You classify a user reply during an active dataframe "
                "normalization review.\n"
                "Understand the intent semantically instead of relying on exact "
                "keyword matches.\n"
                "Return the action in the field `action` and use exactly one of: "
                "PROCEED, EDIT, UNDO, CANCEL, UNKNOWN.\n"
                "Do not invent alternative action names.\n"
                "If the action is EDIT, `extracted_instruction` is required and must not be empty.\n"
                "Observation-only requests about the current dataframe, the current result, "
                "or the outcome of a previous normalization are UNKNOWN, not PROCEED and not EDIT.\n"
                "References to an already applied normalization do not mean the user wants to continue editing the draft."
            ),
        ),
        (
            "user",
            "Current draft preview:\n{draft_preview}\n\nUser reply:\n{user_input}",
        ),
    ]
)
intent_router = intent_router_prompt | llm.with_structured_output(NormalizationIntent)
intent_router_debug_chain = intent_router_prompt | llm
intent_router_repair_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Repair an intent-classifier output for an active dataframe normalization review.\n"
                "Return only canonical values in the field `action`: PROCEED, EDIT, UNDO, CANCEL, UNKNOWN.\n"
                "Preserve the semantic intent of the user reply.\n"
                "If the final action is EDIT, return a short imperative correction in `extracted_instruction`.\n"
                "`extracted_instruction` must not be empty when action is EDIT.\n"
                "Observation-only requests about the current dataframe, current result, or already applied normalization "
                "must resolve to UNKNOWN.\n"
                "If the final action is UNKNOWN, return a helpful clarification in `agent_reply`."
            ),
        ),
        (
            "user",
            (
                "Raw classifier output:\n{raw_intent}\n\n"
                "Current draft preview:\n{draft_preview}\n\n"
                "User reply:\n{user_input}"
            ),
        ),
    ]
)
intent_router_repair = intent_router_repair_prompt | llm.with_structured_output(
    NormalizationIntentResolution
)
intent_router_repair_debug_chain = intent_router_repair_prompt | llm
request_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Decide whether the user is asking to run dataframe "
                "normalization.\n"
                "Normalization means the user wants value-level changes to the dataframe, "
                "such as clustering duplicate labels, standardizing spelling, canonicalizing "
                "values, or rewriting dates or numbers into a consistent format.\n"
                "Differentiate normalization from observation-only requests, general analysis, "
                "inspection of the current dataframe, exploration, or ordinary conversation.\n"
                "Requests to view, inspect, print, explain, summarize, or show the current "
                "dataframe are not normalization unless the user is also asking to change values.\n"
                "A request to show the dataframe after a previous normalization step is still an observation-only request, not a new normalization run.\n"
                "Return the decision in the field `is_normalization_request`.\n"
                "Do not replace that boolean with custom action labels."
            ),
        ),
        (
            "user",
            (
                "Dataset profile:\n{dataset_profile}\n\n"
                "Sample preview:\n{sample_data}\n\n"
                "User request:\n{user_input}"
            ),
        ),
    ]
)
request_router = request_router_prompt | llm.with_structured_output(
    NormalizationRequestDecision
)
request_router_debug_chain = request_router_prompt | llm
request_router_repair_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Repair a normalization-request classifier output.\n"
                "Return only one boolean field: `is_normalization_request`.\n"
                "Infer the boolean semantically from the raw classifier output, "
                "the user request, and the dataframe context."
            ),
        ),
        (
            "user",
            (
                "Raw classifier output:\n{raw_decision}\n\n"
                "Dataset profile:\n{dataset_profile}\n\n"
                "Sample preview:\n{sample_data}\n\n"
                "User request:\n{user_input}"
            ),
        ),
    ]
)
request_router_repair = request_router_repair_prompt | llm.with_structured_output(
    NormalizationRequestResolution
)
request_router_repair_debug_chain = request_router_repair_prompt | llm
scope_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Decide whether the user wants normalization for the whole dataframe or only specific columns.\n"
                "Return exact dataframe column names in the schema field `target_columns`.\n"
                "If the request is broad and applies to the dataset generally, return an empty list.\n"
                "If the request refers to one or more specific fields semantically, map that intent to the exact column names from the available columns.\n"
                "Never invent column names outside the provided dataframe context."
            ),
        ),
        (
            "user",
            (
                "Available columns:\n{available_columns}\n\n"
                "Dataset profile:\n{dataset_profile}\n\n"
                "Sample preview:\n{sample_data}\n\n"
                "User request:\n{user_input}"
            ),
        ),
    ]
)
scope_router = scope_router_prompt | llm.with_structured_output(
    NormalizationScopeDecision
)
scope_router_debug_chain = scope_router_prompt | llm


def _dataframe_to_text(dataframe: pd.DataFrame, rows: int = DEFAULT_SAMPLE_ROWS) -> str:
    preview = dataframe.head(rows)
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return preview.to_string(index=False)


def _profile_series(name: str, series: pd.Series) -> ColumnProfile:
    non_null_values = series.dropna()
    unique_values = non_null_values.astype(str).unique().tolist()
    non_null_count = len(non_null_values)
    unique_count = len(unique_values)
    uniqueness_ratio = unique_count / non_null_count if non_null_count else 0.0
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        non_null_count=non_null_count,
        unique_count=unique_count,
        uniqueness_ratio=round(uniqueness_ratio, 6),
        sample_values=unique_values[:PREVIEW_ITEMS_PER_COLUMN],
    )


def _profile_dataframe(dataframe: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        rows=len(dataframe),
        columns=[
            _profile_series(column_name, dataframe[column_name])
            for column_name in dataframe.columns
        ],
    )


def _dataset_profile_text(dataframe: pd.DataFrame | None) -> str:
    if dataframe is None or dataframe.empty:
        return "No dataframe context available."
    profile = _profile_dataframe(dataframe)
    return json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)


def _sample_preview_text(dataframe: pd.DataFrame | None) -> str:
    if dataframe is None or dataframe.empty:
        return "No dataframe preview available."
    return _dataframe_to_text(dataframe, rows=DEFAULT_SAMPLE_ROWS)


def _available_columns_text(dataframe: pd.DataFrame | None) -> str:
    if dataframe is None or dataframe.empty:
        return "[]"
    return json.dumps(
        [str(column) for column in dataframe.columns], ensure_ascii=False, indent=2
    )


def _chunk_values_for_formatting(values: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_chars = 0
    for value in values:
        value_len = len(value) + 4
        if current_batch and (
            len(current_batch) >= FORMAT_BATCH_MAX_VALUES
            or current_chars + value_len > FORMAT_BATCH_MAX_CHARS
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(value)
        current_chars += value_len
    if current_batch:
        batches.append(current_batch)
    return batches


def _generate_format_mappings(
    *,
    column_name: str,
    series: pd.Series,
    instruction: str,
) -> list[dict[str, Any]]:
    non_null_values = series.dropna()
    if non_null_values.empty:
        return []
    uniques = non_null_values.astype(str).unique().tolist()
    column_profile = _profile_series(column_name, series)
    profile_text = json.dumps(column_profile.model_dump(), ensure_ascii=False, indent=2)
    mappings: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_chunk_values_for_formatting(uniques), start=1):
        payload = {
            "column_name": column_name,
            "column_profile": profile_text,
            "instruction": instruction,
            "uniques": batch,
        }
        result = _invoke_with_debug(
            label=f"apply_format_rule.{column_name}.batch_{batch_index}",
            chain=format_chain,
            debug_chain=format_debug_chain,
            payload=payload,
        )
        mappings.extend(mapping.model_dump() for mapping in result.mappings)
    return mappings


def _resolve_normalization_request_decision(
    decision: NormalizationRequestDecision,
    *,
    dataset_profile: str,
    sample_data: str,
    user_input: str,
) -> bool:
    if decision.is_normalization_request is not None:
        return bool(decision.is_normalization_request)
    repair_payload = {
        "raw_decision": json.dumps(
            decision.model_dump(exclude_none=True), ensure_ascii=False, indent=2
        ),
        "dataset_profile": dataset_profile,
        "sample_data": sample_data,
        "user_input": user_input,
    }
    repaired = _invoke_with_debug(
        label="request_router_repair",
        chain=request_router_repair,
        debug_chain=request_router_repair_debug_chain,
        payload=repair_payload,
    )
    return bool(repaired.is_normalization_request)


def _resolve_normalization_intent(
    intent: NormalizationIntent,
    *,
    draft_preview: str,
    user_input: str,
) -> NormalizationIntentResolution:
    needs_repair = intent.action not in {"PROCEED", "EDIT", "UNDO", "CANCEL", "UNKNOWN"}
    if intent.action == "EDIT" and not intent.extracted_instruction.strip():
        needs_repair = True
    if intent.action == "UNKNOWN" and not intent.agent_reply.strip():
        needs_repair = True
    if not needs_repair:
        return NormalizationIntentResolution(
            action=intent.action,
            extracted_instruction=intent.extracted_instruction,
            agent_reply=intent.agent_reply,
        )
    repair_payload = {
        "raw_intent": json.dumps(
            intent.model_dump(exclude_none=True), ensure_ascii=False, indent=2
        ),
        "draft_preview": draft_preview,
        "user_input": user_input,
    }
    repaired = _invoke_with_debug(
        label="handle_reply.intent_router_repair",
        chain=intent_router_repair,
        debug_chain=intent_router_repair_debug_chain,
        payload=repair_payload,
    )
    if repaired.action == "EDIT" and not repaired.extracted_instruction.strip():
        return NormalizationIntentResolution(
            action="EDIT",
            extracted_instruction=user_input.strip(),
            agent_reply="",
        )
    return repaired


def _active_transforms(values: dict[str, Any]) -> list[dict[str, Any]]:
    manual = values.get("manual_transforms") or []
    auto = values.get("transforms") or []
    return manual or auto


def _apply_transforms_to_dataframe(
    dataframe: pd.DataFrame, transforms: list[dict[str, Any]]
) -> pd.DataFrame:
    result = dataframe.copy()
    for transform in transforms:
        old_col = transform["col_name"]
        new_col = transform["new_name"]
        mapping_dict: dict[str, Any] = {}
        if transform["type"] == "CLUSTER":
            for cluster in transform["clusters"]:
                for variation in cluster["variations"]:
                    mapping_dict[variation] = cluster["golden_key"]
        elif transform["type"] == "FORMAT":
            format_mappings = transform.get("formats") or []
            if not format_mappings and transform.get("instruction"):
                format_mappings = _generate_format_mappings(
                    column_name=old_col,
                    series=result[old_col],
                    instruction=transform["instruction"],
                )
            for mapping in format_mappings:
                mapping_dict[mapping["original"]] = mapping["formatted"]
        result[new_col] = (
            result[old_col].astype(str).map(mapping_dict).fillna(result[old_col])
        )
    return result


def _build_preview_dataframe(
    raw_preview_df: pd.DataFrame,
    normalized_preview_df: pd.DataFrame,
    transforms: list[dict[str, Any]],
) -> pd.DataFrame:
    if raw_preview_df.empty or not transforms:
        return normalized_preview_df
    display_df = pd.DataFrame(index=raw_preview_df.index)
    reference_column = (
        str(raw_preview_df.columns[0]) if len(raw_preview_df.columns) else ""
    )
    if reference_column:
        display_df[reference_column] = raw_preview_df[reference_column]
    for transform in transforms:
        source_column = transform["col_name"]
        target_column = transform["new_name"]
        if (
            source_column in raw_preview_df.columns
            and source_column not in display_df.columns
        ):
            display_df[source_column] = raw_preview_df[source_column]
        preview_target_column = target_column
        if target_column == source_column:
            preview_target_column = f"{target_column} (normalized)"
        if (
            target_column in normalized_preview_df.columns
            and preview_target_column not in display_df.columns
        ):
            display_df[preview_target_column] = normalized_preview_df[target_column]
    return display_df


def _format_transform_preview(transforms: list[dict[str, Any]]) -> str:
    if not transforms:
        return "Подходящих преобразований не найдено."
    sections: list[str] = []
    for transform in transforms:
        block_lines = [
            f"[{transform['col_name']}] -> [{transform['new_name']}] ({transform['type']})"
        ]
        if transform["type"] == "CLUSTER":
            preview_clusters = transform["clusters"][:PREVIEW_ITEMS_PER_COLUMN]
            for cluster in preview_clusters:
                block_lines.append(
                    f"  {cluster['golden_key']} <- {len(cluster['variations'])} вариантов"
                )
            remaining = len(transform["clusters"]) - len(preview_clusters)
            if remaining > 0:
                block_lines.append(f"  ... и еще {remaining} кластеров")
        else:
            preview_formats = transform.get("formats", [])[:PREVIEW_ITEMS_PER_COLUMN]
            if preview_formats:
                for mapping in preview_formats:
                    block_lines.append(
                        f"  {mapping['original']} -> {mapping['formatted']}"
                    )
                remaining = len(transform.get("formats", [])) - len(preview_formats)
                if remaining > 0:
                    block_lines.append(f"  ... и еще {remaining} правил форматирования")
            elif transform.get("instruction"):
                block_lines.append(f"  Правило: {transform['instruction']}")
        sections.append("\n".join(block_lines))
    return "\n\n".join(sections)


def _render_preview_message(state_values: dict[str, Any]) -> str:
    transforms = _active_transforms(state_values)
    raw_df = pd.DataFrame(state_values["raw_records"])
    target_columns = state_values.get("target_columns") or []
    if not transforms:
        return (
            "Не нашел колонок, которые стоит уверенно нормализовать автоматически.\n\n"
            "Можно уточнить цель нормализации естественным языком или отказаться от операции."
        )
    raw_preview_df = raw_df.head(DEFAULT_SAMPLE_ROWS).copy()
    normalized_preview_df = _apply_transforms_to_dataframe(raw_preview_df, transforms)
    preview_df = _build_preview_dataframe(
        raw_preview_df, normalized_preview_df, transforms
    )
    rules_preview = _format_transform_preview(transforms)
    sample_preview = _dataframe_to_text(preview_df, rows=DEFAULT_SAMPLE_ROWS)
    intro = "Предлагаю такой draft нормализации:\n\n"
    if target_columns:
        intro = "Предлагаю такой draft нормализации для выбранных колонок:\n\n"
    return (
        f"{intro}```text\n{rules_preview}\n```\n\n"
        "Пример результата на первых строках:\n"
        f"```text\n{sample_preview}\n```\n\n"
        "Можно подтвердить применение, описать правку своими словами, попросить "
        "откатить последнюю правку или отменить нормализацию."
    )


def profile_dataset(state: NormalizationGraphState) -> dict[str, Any]:
    dataframe = pd.DataFrame(state["raw_records"])
    payload = {
        "dataset_profile": _dataset_profile_text(dataframe),
        "sample_data": state["sample_md"],
    }
    plan = _invoke_with_debug(
        label="profile_dataset.profiler_chain",
        chain=profiler_chain,
        debug_chain=profiler_debug_chain,
        payload=payload,
    )
    return {"plan": [column.model_dump() for column in plan.columns]}


def build_transformations(state: NormalizationGraphState) -> dict[str, Any]:
    dataframe = pd.DataFrame(state["raw_records"])
    transforms: list[dict[str, Any]] = []
    target_columns = set(state.get("target_columns") or [])
    for column in state["plan"]:
        original_name = column["original_name"]
        if column["action_type"] == "PASS" or original_name not in dataframe.columns:
            continue
        if target_columns and original_name not in target_columns:
            continue
        non_null_values = dataframe[original_name].dropna()
        if non_null_values.empty:
            continue
        if column["action_type"] == "CLUSTER":
            uniques = non_null_values.astype(str).unique().tolist()
            column_profile = _profile_series(original_name, dataframe[original_name])
            profile_text = json.dumps(
                column_profile.model_dump(), ensure_ascii=False, indent=2
            )
            payload = {
                "column_name": original_name,
                "column_profile": profile_text,
                "instruction": column["instruction"],
                "uniques": uniques,
            }
            result = _invoke_with_debug(
                label=f"build_transformations.cluster.{original_name}",
                chain=cluster_chain,
                debug_chain=cluster_debug_chain,
                payload=payload,
            )
            transforms.append(
                {
                    "col_name": original_name,
                    "new_name": column["new_name"],
                    "type": "CLUSTER",
                    "instruction": column["instruction"],
                    "clusters": [cluster.model_dump() for cluster in result.clusters],
                    "formats": [],
                }
            )
        elif column["action_type"] == "FORMAT":
            transforms.append(
                {
                    "col_name": original_name,
                    "new_name": column["new_name"],
                    "type": "FORMAT",
                    "instruction": column["instruction"],
                    "clusters": [],
                    "formats": [],
                }
            )
    return {"transforms": transforms}


def apply_transformations(state: NormalizationGraphState) -> dict[str, Any]:
    final_df = _apply_transforms_to_dataframe(
        pd.DataFrame(state["raw_records"]),
        _active_transforms(state),
    )
    return {"final_df": final_df.to_dict(orient="records")}


_builder = StateGraph(NormalizationGraphState)
_builder.add_node("profile_dataset", profile_dataset)
_builder.add_node("build_transformations", build_transformations)
_builder.add_node("apply_transformations", apply_transformations)
_builder.add_edge(START, "profile_dataset")
_builder.add_edge("profile_dataset", "build_transformations")
_builder.add_edge("build_transformations", "apply_transformations")
_builder.add_edge("apply_transformations", END)
_memory = MemorySaver()
normalization_graph = _builder.compile(
    checkpointer=_memory,
    interrupt_before=["apply_transformations"],
)


class DataNormalizationService:
    def __init__(self) -> None:
        self._session_stacks: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _base_config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _current_config(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            stack = self._session_stacks.get(session_id)
            return stack[-1] if stack else None

    def has_pending_session(self, session_id: str) -> bool:
        return self._current_config(session_id) is not None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._session_stacks.pop(session_id, None)

    def _classify_reply_intent(
        self,
        session_id: str,
        user_input: str,
        *,
        label_prefix: str,
    ) -> NormalizationIntentResolution | None:
        current_config = self._current_config(session_id)
        if current_config is None:
            return None
        snapshot = normalization_graph.get_state(current_config)
        draft_preview = _format_transform_preview(_active_transforms(snapshot.values))
        payload = {
            "draft_preview": draft_preview,
            "user_input": user_input,
        }
        raw_intent = _invoke_with_debug(
            label=f"{label_prefix}.intent_router",
            chain=intent_router,
            debug_chain=intent_router_debug_chain,
            payload=payload,
        )
        return _resolve_normalization_intent(
            raw_intent,
            draft_preview=draft_preview,
            user_input=user_input,
        )

    def should_continue_session(self, session_id: str, user_input: str) -> bool:
        if not user_input.strip():
            return False
        intent = self._classify_reply_intent(
            session_id,
            user_input,
            label_prefix="session_gate",
        )
        if intent is None:
            return False
        return intent.action in {"PROCEED", "EDIT", "UNDO", "CANCEL"}

    def _select_target_columns(
        self, user_input: str, dataframe: pd.DataFrame | None = None
    ) -> list[str]:
        if dataframe is None or dataframe.empty or not user_input.strip():
            return []
        payload = {
            "available_columns": _available_columns_text(dataframe),
            "dataset_profile": _dataset_profile_text(dataframe),
            "sample_data": _sample_preview_text(dataframe),
            "user_input": user_input,
        }
        decision = _invoke_with_debug(
            label="start_preview.scope_router",
            chain=scope_router,
            debug_chain=scope_router_debug_chain,
            payload=payload,
        )
        available_columns = {str(column) for column in dataframe.columns}
        selected_columns: list[str] = []
        for column in decision.target_columns:
            if column in available_columns and column not in selected_columns:
                selected_columns.append(column)
        return selected_columns

    def should_start_normalization(
        self, user_input: str, dataframe: pd.DataFrame | None = None
    ) -> bool:
        if not user_input.strip():
            return False
        dataset_profile = _dataset_profile_text(dataframe)
        sample_data = _sample_preview_text(dataframe)
        payload = {
            "dataset_profile": dataset_profile,
            "sample_data": sample_data,
            "user_input": user_input,
        }
        decision = _invoke_with_debug(
            label="should_start_normalization.request_router",
            chain=request_router,
            debug_chain=request_router_debug_chain,
            payload=payload,
        )
        return _resolve_normalization_request_decision(
            decision,
            dataset_profile=dataset_profile,
            sample_data=sample_data,
            user_input=user_input,
        )

    def start_preview(
        self,
        session_id: str,
        dataframe: pd.DataFrame,
        user_input: str = "",
    ) -> NormalizationOutcome:
        self.clear_session(session_id)
        target_columns = self._select_target_columns(user_input, dataframe)
        initial_state: NormalizationGraphState = {
            "raw_records": dataframe.to_dict(orient="records"),
            "sample_md": _dataframe_to_text(dataframe, rows=DEFAULT_SAMPLE_ROWS),
            "target_columns": target_columns,
            "plan": [],
            "transforms": [],
            "manual_transforms": [],
            "final_df": [],
        }
        base_config = self._base_config(session_id)
        for _ in normalization_graph.stream(initial_state, base_config):
            pass
        snapshot = normalization_graph.get_state(base_config)
        if not _active_transforms(snapshot.values):
            self.clear_session(session_id)
            return NormalizationOutcome(
                status="cancelled",
                message=_render_preview_message(snapshot.values),
            )
        with self._lock:
            self._session_stacks[session_id] = [snapshot.config]
        return NormalizationOutcome(
            status="preview",
            message=_render_preview_message(snapshot.values),
        )

    def _preview_from_current_state(
        self, session_id: str, status: Literal["preview", "preview_updated"]
    ) -> NormalizationOutcome:
        current_config = self._current_config(session_id)
        if current_config is None:
            return NormalizationOutcome(
                status="missing_session",
                message="Нет активной сессии нормализации. Запусти ее заново.",
            )
        snapshot = normalization_graph.get_state(current_config)
        return NormalizationOutcome(
            status=status,
            message=_render_preview_message(snapshot.values),
        )

    def edit_session(self, session_id: str, instruction: str) -> NormalizationOutcome:
        current_config = self._current_config(session_id)
        if current_config is None:
            return NormalizationOutcome(
                status="missing_session",
                message="Нет активной сессии нормализации. Запусти ее заново.",
            )
        snapshot = normalization_graph.get_state(current_config)
        payload = {
            "current_state": json.dumps(
                _active_transforms(snapshot.values),
                ensure_ascii=False,
                indent=2,
            ),
            "instruction": instruction,
        }
        updated_result = _invoke_with_debug(
            label="edit_session.update_chain",
            chain=update_chain,
            debug_chain=update_debug_chain,
            payload=payload,
        )
        updated_dicts = [column.model_dump() for column in updated_result.columns]
        new_config = normalization_graph.update_state(
            snapshot.config,
            {"manual_transforms": updated_dicts},
            as_node="build_transformations",
        )
        with self._lock:
            self._session_stacks.setdefault(session_id, []).append(new_config)
        return self._preview_from_current_state(session_id, "preview_updated")

    def undo_session(self, session_id: str) -> NormalizationOutcome:
        with self._lock:
            stack = self._session_stacks.get(session_id)
            if not stack:
                return NormalizationOutcome(
                    status="missing_session",
                    message="Нет активной сессии нормализации. Запусти ее заново.",
                )
            if len(stack) > 1:
                stack.pop()
                prefix = "Последняя правка черновика отменена.\n\n"
            else:
                prefix = "Дальше откатывать некуда.\n\n"
            current_config = stack[-1]
        snapshot = normalization_graph.get_state(current_config)
        return NormalizationOutcome(
            status="preview_updated",
            message=prefix + _render_preview_message(snapshot.values),
        )

    def cancel_session(self, session_id: str) -> NormalizationOutcome:
        self.clear_session(session_id)
        return NormalizationOutcome(
            status="cancelled",
            message="Нормализация отменена. Текущий DataFrame не изменялся.",
        )

    def commit_session(self, session_id: str) -> NormalizationOutcome:
        current_config = self._current_config(session_id)
        if current_config is None:
            return NormalizationOutcome(
                status="missing_session",
                message="Нет активной сессии нормализации. Запусти ее заново.",
            )
        for _ in normalization_graph.stream(None, current_config):
            pass
        final_snapshot = normalization_graph.get_state(self._base_config(session_id))
        final_df = pd.DataFrame(final_snapshot.values.get("final_df", []))
        self.clear_session(session_id)
        result_preview = _dataframe_to_text(final_df, rows=DEFAULT_SAMPLE_ROWS)
        return NormalizationOutcome(
            status="committed",
            message=(
                "Нормализация применена к текущему DataFrame.\n\n"
                "Первые строки результата:\n"
                f"{result_preview}"
            ),
            dataframe=final_df,
        )

    def handle_reply(self, session_id: str, user_input: str) -> NormalizationOutcome:
        current_config = self._current_config(session_id)
        if current_config is None:
            return NormalizationOutcome(
                status="missing_session",
                message="Нет активной сессии нормализации. Запусти ее заново.",
            )
        intent = self._classify_reply_intent(
            session_id,
            user_input,
            label_prefix="handle_reply",
        )
        if intent is None:
            return NormalizationOutcome(
                status="missing_session",
                message="РќРµС‚ Р°РєС‚РёРІРЅРѕР№ СЃРµСЃСЃРёРё РЅРѕСЂРјР°Р»РёР·Р°С†РёРё. Р—Р°РїСѓСЃС‚Рё РµРµ Р·Р°РЅРѕРІРѕ.",
            )
        if intent.action == "PROCEED":
            return self.commit_session(session_id)
        if intent.action == "EDIT":
            return self.edit_session(session_id, intent.extracted_instruction)
        if intent.action == "UNDO":
            return self.undo_session(session_id)
        if intent.action == "CANCEL":
            return self.cancel_session(session_id)
        return NormalizationOutcome(status="preview", message=intent.agent_reply)


normalization_service = DataNormalizationService()
