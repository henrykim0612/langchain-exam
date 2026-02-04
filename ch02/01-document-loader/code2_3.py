# PDF 문서 추출
# pypdf 설치 필요

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./test.pdf")
pages = loader.load()

print(pages)