# 가상 문서 임베딩

from langchain_core.runnables import chain

from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector(
    connection=connection,
    collection_name="langchain",
    embeddings=embeddings_model
)

retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_hyde = ChatPromptTemplate.from_template(
    '''
질문에 답할 구절을 작성해 주세요.
질문: {question}

구절:''')

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # 가짜 문서가 이상하면 망하기 때문에 창의성 배제

generate_doc = (prompt_hyde | llm | StrOutputParser())

# 가상 문서를 retriever 의 입력으로 전달해 임베딩 생성
# 그리고 벡터 저장소에서 유사한 문서를 검색
retrieval_chain = generate_doc | retriever

# 마지막으로 검색한 문서를 최종 프롬프트에 컨텍스트로 전달해 LLM이 출력을 생성하도록 지시
prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'


@chain
def qa(input):
    docs = retrieval_chain.invoke(input)
    print("docs:")
    print(docs)
    formatted = prompt.invoke({'context': docs, 'question': input})
    answer = llm.invoke(formatted)
    return answer


result = qa.invoke(query)
print(result.content)
