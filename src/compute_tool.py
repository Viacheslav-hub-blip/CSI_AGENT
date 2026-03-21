
class MyRunner:
    def __init__(self, shell):
        self.shell = shell  # e.g. IPython.get_ipython()

    async def run_code(self, code: str, source_dataframe, prompt, GigaChat_Max, show_in_notebook: bool = False) -> Dict[str, Any]:
        """
        Выполняет code через shell.run_cell_async и:
        - перехватывает stdout в stdout_text
        - перехватывает warnings / stderr в stderr_text
        - перехватывает traceback и помещает его в stderr_text
        - если show_in_notebook==False — traceback НЕ отображается в Jupyter (только в stderr_text)
        - если show_in_notebook==True — дополнительно вызывает оригинальный showtraceback
        """
        logger.info('##RUN CODE##', extra = {'extra_info': ''})
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        shell = self.shell

        orig_showtraceback = getattr(shell, "showtraceback", None)

        def _capture_showtraceback(exc_tuple=None, *a, **kw):
            # Получаем exc_tuple (etype, evalue, etb)
            if exc_tuple is None:
                exc_tuple = sys.exc_info()
            etype, evalue, etb = exc_tuple
            tb_text = "".join(traceback.format_exception(etype, evalue, etb))
            # Сохраняем в наш буфер
            err_buf.write(tb_text)

            # Если пользователь захотел видеть traceback в notebook — вызываем оригинал
            if show_in_notebook and orig_showtraceback:
                try:
                    orig_showtraceback(exc_tuple, *a, **kw)
                except Exception:
                    # Защита: если оригинал падает, мы уже записали traceback
                    pass
            # Иначе — НЕ вызываем оригинал => в notebook ничего не показываем

        exec_result = None
        try:
            # Подменяем поведение отображения traceback
            shell.showtraceback = _capture_showtraceback
            shell.user_ns.update(locals())
            # Перенаправляем stdout/stderr (warnings обычно идут в stderr)
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                exec_result = await shell.run_cell_async(code, store_history=False)
        finally:
            # Восстанавливаем оригинал обязательно
            if orig_showtraceback is not None:
                shell.showtraceback = orig_showtraceback
            else:
                try:
                    delattr(shell, "showtraceback")
                except Exception:
                    pass

        stdout_text = out_buf.getvalue()
        stderr_text = err_buf.getvalue()
        stderr_text = stderr_text.replace('{', '').replace('}', '')

        logger.info('##RUN CODE## текст логов (ошибки)', extra = {'extra_info': f'Логи: {stderr_text}'})

        promt_error_detected = ChatPromptTemplate.from_messages([
            ("system", prompt_to_parsing_log),
            ("human", "{user_query}")
        ])

        chain = promt_error_detected | GigaChat_Max
        response = await chain.ainvoke({"user_query": stderr_text})
        logger.info('##RUN CODE## заключение модели об ошибке на основе логов', extra = {'extra_info': f'Ответ модели: {response.content.lower()}'})

        if "success" in response.content.lower():
            error_detected = False
        else:
            error_detected = True

        return {
            "success": not error_detected,
            "stdout": stdout_text,
            "stderr": stderr_text,  # тут — и warnings, и traceback (если был)
            "exec_result": exec_result,
        }

# # Код агента

# In[1]:

class State(TypedDict):
    user_input: str
    generated_prompt: str
    generated_python_code_with_llm: str
    generated_python_code_without_llm: str
    code_exec_error: str
    attempt: int
    result_status: str

# In[ ]:

class ProcessDataFrameAgent():
    def __init__(self, df: pd.DataFrame, llm_model):
        self.state = State
        self.df = df
        self.app = self.compile_graph()
        self.python_executor = MyRunner(get_ipython())
        self.llm_model = llm_model

    async def _prompt_template(self, values: dict) -> str:
        ''' Шаблон prompt запроса, который будет использоваться для обработки записей (извлечения сущностей)
        args: values - словарь с значениями для prompt запроса
        '''
        prompt = f"""
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
<instructions>
    {values["instructions"]}
</instructions>
"""
        return prompt

    async def determining_need_use_llm(self, state: State):
        ''' Определеляет, нужно ли использовать LLM для вычисления значений новой колонки
        returns: str - да/нет
        '''
        logger.info('##determining_need_use_llm##', extra = {'extra_info': f''})
        prompt = ChatPromptTemplate.from_messages([
            ("system",prompt_determine_need_llm + "<data_sctructure>" + self.df.head(1).to_xml()),
            ('human', "{user_query}")
        ])
        chain = prompt | self.llm_model
        print("prompt", prompt)
        response = await chain.ainvoke(
            {
                "user_query": state['user_input']
            }
        )
        logger.info('##determining_need_use_llm## answer', extra = {'extra_info': f'{response.content}'})
        return Command(
            goto="generate_prompt",
        )
        # return Command(
        #     goto="generate_python_code_without_llm",
        # )

    async def generate_prompt(self, state: State):
        '''Узел генерации prompt запроса дял извлечения сущностей'''
        logger.info('##GENERATE_PROMPT##', extra = {'extra_info': f''})
        prompt = ChatPromptTemplate.from_messages([
            ("system",prompt_for_generate_prompt),
            ('human', "{user_query}")
        ])
        chain = prompt | self.llm_model
        response = await chain.ainvoke(
            {
                "user_query": state['user_input']
            }
        )
        response_dict = eval(response.content)
        prompt_for_parsing = await self._prompt_template(response_dict)

        logger.info('##GENERATE_PROMPT## сгенерированный prompt запрос для извлечения', extra = {'extra_info': f'Сгенерированный prompt запрос для извлечения: {prompt_for_parsing}'})
        return Command(
            goto="generate_python_code_with_giga",
            update={"generated_prompt":prompt_for_parsing}
        )

    async def generate_python_code_with_llm(self, state: State):
        '''Узел генерации кода для вычисления значений с использованием GigaChat'''
        global prompt_generate_python_func_with_giga
        logger.info('##GENERATE_PYRTHON_CODE_WITH_GIGA##', extra = {'extra_info': f''})
        try:
            exected_code = state['generated_python_code_with_llm']
            error_from_previos_step = state['code_exec_error']
        except:
            exected_code = 'Код еще не генерировался'
            error_from_previos_step = 'Код еще не выполнялся'

        prompt_generate_python_func_with_giga = prompt_generate_python_func_with_giga.replace("data_structure_replaced", self.df.head(1).to_xml())

        logger.info('##GENERATE_PYRTHON_CODE_WITH_GIGA## prompt для генерации Python кода', extra = {'extra_info': f'PROMPT запрос для генерации Python code {prompt_generate_python_func_with_giga}'})
        prompt_python = ChatPromptTemplate.from_messages([
            ("system",prompt_generate_python_func_with_giga),
            ("human", """Запрос пользователя: {user_query}. Предыдущий сгенерированный код: {exected_code}. Ошибка выполнения предыдущего сгенерированного кода: {error_from_previos_step}""")
        ])
        chain_python = prompt_python| self.llm_model
        response = await chain_python.ainvoke(
            {
                "user_query": state['user_input'],
                "exected_code": exected_code,
                "error_from_previos_step": error_from_previos_step
            }
        )
        code = response.content.replace('```', '').replace('python','')
        logger.info('##GENERATE_PYRTHON_CODE_WITH_GIGA## Сгенерированный код', extra = {'extra_info': f'Сгенерированный Python code {code}'})
        return Command(
            goto="execute_python_code",
            update={"generated_python_code_with_llm":code}
        )

    async def generate_python_code_without_llm(self, state: State):
        '''Генерируй код, который извлекает значения без использования LLM'''
        global prompt_generate_python_func_without_giga
        try:
            exected_code = state['generated_python_code_without_llm']
            error_from_previos_step = state['code_exec_error']
        except:
            exected_code = 'Код еще не генерировался'
            error_from_previos_step = 'Код еще не выполнялся'

        prompt_generate_python_func_without_giga = prompt_generate_python_func_without_giga.replace(
            "data_structure_replaced",
            (self.df.head(1).to_xml() + "<data_types>" + ", ".join([f"{col}: {dtype}" for col, dtype in self.df.dtypes.items()]) + "/<data_types>" + "<empty_values>: (True / False):" + ", ".join([f"{col}: {self.df[col].isna().any()}" for col in self.df.columns]))
        )
        logger.info('##GENERATE_PYRTHON_CODE_WITH_GIGA## prompt для генерации Python кода', extra = {'extra_info': f'PROMPT запрос для генерации Python code {prompt_generate_python_func_without_giga}'})
        logger.info('##generate_python_code_without_llm##', extra = {'extra_info': f''})
        prompt_python = ChatPromptTemplate.from_messages([
            ("system",prompt_generate_python_func_without_giga),
            ("human", """Запрос пользователя: {user_query}. Предыдущий сгенерированный код: {exected_code}. Ошибка выполнения предыдущего сгенерированного кода: {error_from_previos_step}""")
        ])
        chain_python = prompt_python| self.llm_model
        response = await chain_python.ainvoke(
            {
                "user_query": state['user_input'],
                "exected_code": exected_code,
                "error_from_previos_step": error_from_previos_step
            }
        )
        code = response.content.replace('```', '').replace('python','')
        logger.info('##generate_python_code_without_llm## Сгенерированный код', extra = {'extra_info': f'Сгенерированный Python code {code}'})
        return Command(
            goto="execute_python_code",
            update={"generated_python_code_without_llm":code}
        )

    async def execute_python_code(self, state: State):
        '''Узел для выполнения Python кода'''
        logger.info('##EXECUTE_PYTHON_CODE##', extra = {'extra_info': f''})
        GigaChat_Max = self.llm_model
        source_dataframe = self.df
        code = state['generated_python_code_with_llm'] if state.get('generated_python_code_with_llm', False) else state['generated_python_code_without_llm']
        generated_prompt = state.get('generated_prompt', '')
        exec_res = await self.python_executor.run_code(code, self.df, generated_prompt, GigaChat_Max)
        logger.info('##EXECUTE_PYTHON_CODE## exec_res', extra = {'extra_info': exec_res})
        if exec_res['success'] == True:
            return {"result_status": "done"}
        else:
            if state.get('generated_python_code_with_llm', False):
                return Command(
                    goto="generate_python_code_with_llm",
                    update={
                        "code_exec_error":exec_res['stderr'],
                        "attempt": state.get("attempt", 0) + 1
                    }
                )
            else:
                return Command(
                    goto="generate_python_code_without_llm",
                    update={
                        "code_exec_error":exec_res['stderr'],
                        "attempt": state.get("attempt", 0) + 1
                    }
                )

    def compile_graph(self):
        workflow = StateGraph(self.state)
        workflow.add_node("determining_need_use_llm", self.determining_need_use_llm)
        workflow.add_node("generate_prompt", self.generate_prompt)
        workflow.add_node("generate_python_code_with_llm", self.generate_python_code_with_llm)
        workflow.add_node("generate_python_code_without_llm", self.generate_python_code_without_llm)
        workflow.add_node("execute_python_code", self.execute_python_code)

        workflow.add_edge(START, "determining_need_use_llm")
        workflow.add_edge("execute_python_code", END)

        return workflow.compile()

    async def __call__(self, user_input:str):
        return await self.app.ainvoke({"user_input": user_input})

    @staticmethod
    def _extract_token_text(event: dict[str, Any]) -> str:
        data = event.get("data", {})
        chunk = data.get("chunk")
        if chunk is None:
            return ""
        if hasattr(chunk, "content"):
            content = chunk.content
            try:
                json_content = eval(content)
                return json_content["though"]
            except Exception as e:
                return chunk.content
        if isinstance(chunk, str):
            return chunk
        return ""

    async def run_with_streaming(
        self,
        inputs
    ):
        '''Запуск агентом с потокой передачей ответов'''
        last_step = ""
        was_generate_prompt = False
        async for event in self.app.astream_events(inputs):
            kind_event = event.get("event", "")
            metadata = event.get("metadata", "")
            graph_node = metadata.get("langgraph_node", "")
            if graph_node == "generate_prompt" and last_step != "generate_prompt":
                last_step = "generate_prompt"
                if was_generate_prompt == False:
                    was_generate_prompt = True
                    yield "Generating prompt query..."
                else:
                    yield "Error execute Python code. Retry generation..."
            if graph_node == "determining_need_use_llm" and last_step != "determining_need_use_llm":
                last_step = "determining_need_use_llm"
                yield "Datermining to need use llm"
            if graph_node == "generate_python_code_with_llm" and last_step != "generate_python_code_with_llm":
                last_step = "generate_python_code_with_llm"
                yield "Generating Python code with using LLM..."
            if graph_node == "generated_python_code_without_llm" and last_step != "generated_python_code_without_llm":
                last_step = "generated_python_code_without_llm"
                yield "Generating Python code..."
            if graph_node == "execute_python_code" and last_step != "execute_python_code":
                last_step = "execute_python_code"
                yield "Executing Python code..."
