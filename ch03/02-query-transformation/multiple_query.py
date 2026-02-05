# 다중 쿼리 구성
from langchain_core.runnables import chain
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

retriever = db.as_retriever(search_kwargs={"k": 5})

# 다중 쿼리를 위한 프롬프트
perspectives_prompt = ChatPromptTemplate.from_template(
    '''
    당신은 AI 언어 모델 어시스턴트입니다. 주어진 사용자 질문의 다섯 가지 버전을 생성하여 벡터 데이터베이스에서 관련 문서를 검색하세요. 
    사용자 질문에 대한 다양한 관점을 생성함으로써 사용자가 거리 기반 유사도 검색의 한계를 극복할 수 있도록 돕는 것이 목표입니다. 
    이러한 대체 질문을 개행으로 구분하여 제공하세요. 
    원래 질문: {question}''')

llm = ChatOpenAI(model="gpt-5-nano")


def parse_queries_output(message):
    print("다중 쿼리 질문:")
    print(message)
    return message.content.split('\n')


query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    print(document_lists)
    # 목록 여러 개를 포함한 리스트를 평탄화하고 중복 제거
    deduped_docs = {doc.page_content: doc for sublist in document_lists for doc in sublist}
    # 고유한 문서만 반환
    return list(deduped_docs.values())


retriever_chain = query_gen | retriever.batch | get_unique_union

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'


@chain
def multi_query_qa(input):
    # 관련 문서 검색
    docs = retriever_chain.invoke(input)
    formatted = prompt.invoke({'context': docs, 'question': input})
    answer = llm.invoke(formatted)
    return answer


result = multi_query_qa.invoke(query)
print(result.content)
