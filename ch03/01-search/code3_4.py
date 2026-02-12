# 관련 문서를 조회한 후, 마지막으로 해당 문서를 원본 프롬프트의 컨텍스트에 추가한 다음 LLM을 호출해 최종 출력을 생성(캡슐화 방식)

from langchain_core.runnables import chain

from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector(
    connection=connection,
    collection_name="langchain",
    embeddings=embeddings_model
)

retriever = db.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # temperature=0: 창의성 배제


@chain
def qa(input):
    # 관련 문서 검색
    docs = retriever.invoke(input)
    # 프롬프트 포매팅
    formatted = prompt.invoke({"context": docs, "question": input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return {"answer": answer, "docs": docs}


# 실행
query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

result = qa.invoke(query)
print(result)
