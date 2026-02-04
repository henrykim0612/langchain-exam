# 웹페이지 추출
# beautifulsoup4 설치 필요

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com")
docs = loader.load()

print(docs)