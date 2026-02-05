# PGVector 를 활용한 임베딩 저장

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

# 도커 연결 설정
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

# 문서 로드 후 분할
raw_documents = TextLoader("./test.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# 문서에 대한 임베딩 생성
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 저장
db = PGVector.from_documents(documents, embeddings_model, connection=connection)