# 선언형 구성 예시(스트리밍 호출)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 구성요소
template = ChatPromptTemplate.from_messages([
    ('system', '당신은 친절한 어시스턴트입니다.'),
    ('human', '{question}')
])

model = ChatOpenAI(model='gpt-5-mini')

# 연산자 | 로 결합한다
chatbot = template | model

for part in chatbot.stream({'question': '거대 언어 모델은 어디서 제공하나요?'}):
    print(part)
