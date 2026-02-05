# 관련 문서를 조회한 후, 마지막으로 해당 문서를 원본 프롬프트의 컨텍스트에 추가한 다음 LLM을 호출해 최종 출력을 생성

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

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

retriever = db.as_retriever(search_kwargs={"k": 2})

docs = retriever.invoke(query)

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # temperature=0: 창의성 배제
llm_chain = prompt | llm

# 관련 문서를 사용한 답변
result = llm_chain.invoke({"context": docs, "question": query})

print(result)
