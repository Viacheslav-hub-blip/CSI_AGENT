prompt_to_parsing_log = """
You are checking Python execution logs.
Return only one word:
- `success` if the log does not contain a real execution error;
- `error` if the code failed.
Do not add explanations.
""".strip()
prompt_determine_need_llm = """
Decide whether the user's task requires an LLM-based semantic extraction step.
Return only one word:
- `yes` if the task needs understanding free text, entity extraction, normalization from messy text, or row-by-row semantic interpretation;
- `no` if the task can be solved with deterministic pandas/python operations only.
""".strip()
prompt_for_generate_prompt = """
You are preparing a row-level extraction prompt for another model.
Return only valid JSON with exactly these string keys:
- `agent_role`
- `instructions`
- `important`
- `input_format`
- `output_format`
- `available_values`
- `examples`
The JSON must describe how to extract the value requested by the user from one dataframe row.
Do not wrap the JSON in markdown fences.
""".strip()
prompt_generate_python_func_with_giga = """
Generate only Python code, without markdown fences.
Task:
- write code that creates `final_df` from `source_dataframe.copy()`;
- use the provided string variable `prompt` as a system instruction for row-level extraction;
- use the provided model object `llm` (or `GigaChat_Max`) to process rows that need semantic understanding;
- store the final dataframe in the variable `final_df`.
Requirements:
- never read files from disk;
- work only with variables already available in memory;
- keep the code deterministic apart from LLM calls;
- if you create a new column, use a clear snake_case name;
- prefer the simplest possible code structure;
- process rows sequentially with a normal `for` loop when row-wise handling is needed;
- do not use asynchronous row processing;
- do not use `async_process_rows_limited(...)` or `run_async_tasks_limited(...)`;
- avoid extra helper functions unless they are truly necessary;
- before major stages, call `print_status("...")` with a short human-readable message;
- if you iterate through rows or batches, call `report_progress(processed_rows, total_rows, message)` periodically;
- call `report_progress(0, total_rows, "...")` near the start of row-wise work and a final progress update at the end;
- if the task is impossible, still return a safe dataframe copy and add a comment in code.
- if `Previous execution error` is provided, you must use it to fix the code instead of repeating the same approach;
- if `Previous stdout` is provided, use it as an execution hint;
- on retry, change the failing part of the code explicitly.
Important anti-pattern to avoid:
- never compute one fallback value once and copy it into all rows;
- never wrap the whole dataframe processing in one broad `try/except` that returns the same value for every row after one failure;
- if one row fails, set a fallback only for that row and continue processing the remaining rows.
Good example for row-wise LLM extraction:
1. Build `rows = source_dataframe.to_dict("records")`
2. Create an empty list like `results = []`
3. Loop through rows with a normal `for row in rows:`
4. Inside the loop, call the model for this specific `row`
4. If parsing fails for this row, return `""` or `None` only for this row
5. Append one result per row to `results`
6. Assign `final_df["new_column"] = results`
Bad example:
- call the model once;
- if that one call fails, fill `final_df["new_column"] = "error_value"` for all rows.
Good example for per-row fallback:
- `if row_result is None: return ""`
- `return row_result`
- then append that single-row result into the list of results
Good code example:
`rows = source_dataframe.to_dict("records")`
`final_df = source_dataframe.copy()`
`results = []`
`for index, row in enumerate(rows, start=1):`
`    try:`
`        row_text = json.dumps(row, ensure_ascii=False)`
`        response = llm.invoke(prompt + "\\n\\nRow:\\n" + row_text)`
`        results.append(str(response.content).strip())`
`    except Exception:`
`        results.append("")`
`    if index % 5 == 0 or index == len(rows):`
`        report_progress(index, len(rows), "Извлекаю значения по строкам")`
`final_df["new_column"] = results`
Good retry examples:
- if the previous error is `NameError`, define the missing variable before using it;
- if the previous error is `ValueError: Length of values does not match length of index`, make sure the output list length equals `len(source_dataframe)`;
- if the previous error is `AttributeError` on a dataframe column, check the actual column names from the provided schema and use the correct one;
- if the previous error happened inside row processing, keep the error local to that row and do not replace the entire column with one fallback value.
Current data structure:
data_structure_replaced
""".strip()
prompt_generate_python_func_without_giga = """
Generate only Python code, without markdown fences.
Task:
- solve the user request with pandas/python only;
- create `final_df = source_dataframe.copy()`;
- apply the transformation or calculation to `final_df`;
- keep the result in the variable `final_df`.
Requirements:
- do not use external files or network;
- do not use markdown;
- prefer vectorized pandas code;
- if a new column is created, use a clear snake_case name;
- prefer the simplest possible code;
- when row-wise processing is needed, use a normal sequential loop;
- do not use asynchronous code or async helpers;
- avoid unnecessary helper functions, wrappers, or abstractions;
- before major stages, call `print_status("...")` with a short human-readable message;
- if you process rows in a loop, call `report_progress(processed_rows, total_rows, message)` periodically;
- if the request is a filter or reshape, `final_df` should contain the transformed dataframe.
- if `Previous execution error` is provided, use it to fix the failing code path and do not repeat the same mistake;
- if `Previous stdout` is provided, use it as a debugging hint.
Important anti-pattern to avoid:
- never assign one constant fallback value to the whole output column because one row could not be processed;
- if processing is row-wise, produce one result per input row;
- handle row errors locally instead of replacing the entire result with a single default.
Good example:
- prepare a list of per-row outputs;
- if one row has bad data, return a fallback only for that row;
- after processing, assign the collected list to the target column.
Good code example:
`final_df = source_dataframe.copy()`
`results = []`
`for index, (_, row) in enumerate(final_df.iterrows(), start=1):`
`    try:`
`        results.append(str(row["target_column"]).strip())`
`    except Exception:`
`        results.append("")`
`    if index % 10 == 0 or index == len(final_df):`
`        report_progress(index, len(final_df), "Обрабатываю строки последовательно")`
`final_df["normalized_value"] = results`
Good retry examples:
- if a column name is wrong, replace it with an existing column from the schema;
- if a list has the wrong length, rebuild it so it has exactly one value per input row;
- if code tries to use a variable before assignment, initialize it before the loop or before the assignment.
Bad example:
- catch one exception outside the loop and then do `final_df["new_column"] = "fallback"` for every row.
Current data structure:
data_structure_replaced
""".strip()
