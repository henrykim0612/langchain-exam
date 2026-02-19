# 의미론적 라우팅

from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

physics_template = '''당신은 매우 똑똑한 물리학 교수입니다. 
    당신은 물리학에 대한 질문에 간결하고 쉽게 이해할 수 있는 방식으로 대답하는 데 뛰어납니다.
    당신이 질문에 대한 답을 모를 때는 모른다고 인정합니다.
    다음 질문에 답하세요.: {query}
    '''
math_template = '''
    당신은 매우 뛰어난 수학자입니다. 당신은 수학 문제에 답하는 데 뛰어납니다.
    당신은 어려운 문제를 구성 요소로 분해하고 구성 요소를 해결한 다음
    함께 모아 더 넓은 질문에 대답합니다.
    다음 질문에 답하세요.: {query}
    '''

# 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


# 질문을 프롬프트에 라우팅

@chain
def prompt_router(query):
    query_embeddings = embeddings.embed_query(query)
    similarity = cosine_similarity([query_embeddings], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print('수학 프롬프트 사용' if most_similar == math_template else '물리 프롬프트 사용')
    return PromptTemplate.from_template(most_similar)


semantic_router = (prompt_router | ChatOpenAI(model='gpt-5-mini') | StrOutputParser())

result = semantic_router.invoke('상대성 이론이란 무엇인가요?')
print('\n의미론적 라우팅 결과: ', result)
