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

prompt_rag_fusion = ChatPromptTemplate.from_template(
    '''
하나의 입력 쿼리를 기반으로 여러 개의 검색 쿼리를 생성하는 유용한 어시스턴트입니다.
다음과 관련된 여러 검색 쿼리를 개행으로 구분하여 영문으로 생성합니다: 
{question}

출력(쿼리 4개):
''')


def parse_queries_output(message):
    print("다중 쿼리 결과:")
    print(message)
    return message.content.split('\n')


llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

query_gen = prompt_rag_fusion | llm | parse_queries_output


# 문서를 재정렬하는 RRF 알고리즘
def reciprocal_rank_fusion(results: list[list], k=60):
    '''여러 순위 문서 목록에 대한 상호 순위 융합 및 RRF 공식에 사용되는 선택적 매개변수 k 입니다.'''
    # 여기서 k 값은 랭킹 점수의 차이를 완만하게 만들어주는 완충 장치(k 값이 작으면 1등 문서 점수가 거의 압도적)
    # 사전을 초기화해 각 문서에 대한 융합된 점수를 보관
    # 고유성을 보장하기 위해 문서가 콘텐츠별로 키를 생성
    fused_scores = {}
    documents = {}
    for docs in results:
        # 목록에 있는 각 문서를 순위(목록 내 위치)에 따라 반복
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    # 융합된 점수를 기준으로 문서를 내림차순으로 정렬하여 최종 재순위 결과를 정리
    reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retriever_chain = query_gen | retriever.batch | reciprocal_rank_fusion

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'


@chain
def rag_fusion(input):
    docs = retriever_chain.invoke(input)
    print("docs:")
    print(docs)
    formatted = prompt.invoke({'context': docs, 'question': input})
    answer = llm.invoke(formatted)
    return answer


result = rag_fusion.invoke(query)
print("결과:")
print(result.content)
