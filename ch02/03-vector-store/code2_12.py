# PGVector 를 활용한 문서 삭제

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
import uuid

# 도커 연결 설정
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector(
    connection=connection,
    collection_name="langchain",
    embeddings=embeddings_model
)

db.delete(ids=['442c91b0-42a5-40cd-8fbf-43b24f7c3404', 'ef68e322-52e5-46fd-af01-6ab9f10b0ced'])
