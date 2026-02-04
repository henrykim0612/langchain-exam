# PGVector 를 활용한 문서 검색

from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# 도커 연결 설정
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector(
    connection=connection,
    collection_name="langchain",
    embeddings=embeddings_model
)

results = db.similarity_search('pond', k=4)
print(results)
