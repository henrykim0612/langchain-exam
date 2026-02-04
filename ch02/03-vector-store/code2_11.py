# PGVector 를 활용한 문서 추가

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

ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [Document(page_content="there are cats in the pond", metadata={'location': 'pond', 'topic': 'animals'}),
     Document(page_content="ducks are also found in the pond", metadata={'location': 'pond', 'topic': 'animals'})],
    ids=ids)
