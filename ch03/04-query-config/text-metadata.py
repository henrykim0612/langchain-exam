# pip install lark

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

docs = [
    Document(
        page_content='과학자들이 공룡을 되살리고 대혼란이 일어난다.',
        metadata={'year': 1993, 'rating': 7.7, 'genre': 'SF'},
    ),
    Document(
        page_content='레오나르도 디카프리오가 꿈속의 꿈속의 꿈속의 꿈속에 빠진다.',
        metadata={'year': 2010, 'director': '크리스토퍼 놀란', 'rating': 8.2},
    ),
    Document(
        page_content='심리학자인 형사가 꿈속의 꿈속의 꿈속의 꿈속의 꿈속에 빠진다. 인셉션이 이 발상을 차용했다.',
        metadata={'year': 2006, 'director': '곤 사토시', 'rating': 8.6},
    ),
    Document(
        page_content='평범한 체형의 매우 건강하고 순수한 매력을 지닌 여성들을 남성들이 동경한다.',
        metadata={'year': 2019, 'director': '그레타 거윅', 'rating': 8.3},
    ),
    Document(
        page_content='장난감들이 살아 움직이며 신나는 시간을 보낸다',
        metadata={'year': 1995, 'genre': '애니메이션'},
    ),
    Document(
        page_content='세 남자가 구역으로 들어가고, 세 남자가 구역 밖으로 나온다.',
        metadata={
            'year': 1979,
            'director': '안드레이 타르코프스키',
            'genre': '스릴러',
            'rating': 9.9,
        },
    ),
]

# 문서 임베딩 생성성
embeddings_model = OpenAIEmbeddings()

vectorstore = PGVector.from_documents(
    docs, embeddings_model, connection=connection)

# 쿼리 필드 생성
fields = [
    AttributeInfo(
        name='genre',
        description='영화 장르',
        type='string or list[string]',
    ),
    AttributeInfo(
        name='year',
        description='영화 개봉 연도',
        type='integer',
    ),
    AttributeInfo(
        name='director',
        description='영화 감독',
        type='string',
    ),
    AttributeInfo(
        name='rating',
        description='영화 평점 1-10점',
        type='float',
    ),
]

description = '영화에 대한 간략한 정보'
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, description, fields)

# 필터 적용
print(retriever.invoke('평점이 8.5점 이상인 영화가 보고 싶어요.'))

print('\n')

# 다양한 필터 적용
print(retriever.invoke(
    '평점이 높은(8.5점 이상) SF영화는 무엇인가요?'))
