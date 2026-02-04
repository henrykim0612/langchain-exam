# 관련 문서 검색

from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector(
    connection=connection,
    collection_name="langchain",
    embeddings=embeddings_model
)

# 백터 저장소에서 관련 문서 검색
retriever = db.as_retriever(search_kwargs={"k": 2})

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

print(retriever.invoke(query))
