from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="minimax/minimax-m2.5",
    # model="google/gemini-2.5-flash",
    # google/gemini-3-pro-preview
    # google/gemini-2.5-flash

    # model="kwaipilot/kat-coder-pro:free",
    temperature=0.7
)

embeddings = OpenAIEmbeddings(
    # 1. Меняем базовый URL на OpenRouter
    openai_api_base="https://openrouter.ai/api/v1",

    # 2. Передаем ключ OpenRouter
    openai_api_key="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",

    # 3. Указываем модель (OpenRouter требует указывать провайдера, например 'openai/')
    model="openai/text-embedding-3-small",

    # Опционально: отключаем проверку SSL, если возникают странные ошибки сети
    # check_embeddings=True
)


def get_answer(prompt: str, model, prompt_params: dict = None) -> str:
    if prompt_params is None:
        prompt_params = {}
    chain = ChatPromptTemplate.from_template(prompt) | model | StrOutputParser()
    return chain.invoke(prompt_params)


if __name__ == "__main__":
    prompt = ChatPromptTemplate.from_template("")
    answer = get_answer(prompt)
    print(answer)
