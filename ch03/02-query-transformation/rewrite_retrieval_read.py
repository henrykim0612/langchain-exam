# 쿼리 재작성 프롬프트를 사용한 호출

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

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트: {context}

질문: {question}
''')

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # temperature=0: 창의성 배제

rewrite_prompt = ChatPromptTemplate.from_template(
    '''
    웹 검색 엔진이 주어진 질문에 답할 수 있도록 더 나은 영문 검색어를 제공하세요. 쿼리는 \'**\'로 끝내세요.
    
    질문: {x}
    
    답변:
''')


def parse_rewriter_output(message):
    print("치환 전:")
    print(message)
    return message.content.strip('\'').strip('**')


rewriter = rewrite_prompt | llm | parse_rewriter_output


@chain
def qa_rrr(input):
    # 쿼리 재작성
    new_query = rewriter.invoke(input)
    print('재작성한 쿼리:')
    print(new_query)
    # 관련 문서 검색
    docs = retriever.invoke(new_query)
    # 프롬프트 포매팅
    print('docs:')
    print(docs)
    formatted = prompt.invoke({"context": docs, "question": input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return answer


query = '일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜박했네요. 고대 그리스 철학사의 주요 인물은 누구인가요?'

result = qa_rrr.invoke(query)
print(result)
