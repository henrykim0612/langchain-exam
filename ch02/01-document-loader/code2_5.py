# 코드의 분할

from langchain_text_splitters import (RecursiveCharacterTextSplitter, Language)

PYTHON_CODE = '''
def hello_world():
    print("Hello, world!")
    
# Call the function
hello_world()
'''

python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=50, chunk_overlap=0)
python_docs = python_splitter.create_documents([PYTHON_CODE])

print(python_docs)
