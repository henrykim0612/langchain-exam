# 대화 기록을 활용한 메모리 시스템

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ('system', '당신은 친절한 어시스턴트입니다. 모든 질문에 최선을 다해 답하세요.'),
    ('placeholder', '{messages}'),
])

model = ChatOpenAI(model='gpt-5-mini')

chain = prompt | model

response = chain.invoke({
    'messages': [
        ('human', '다음 한국어 문장을 프랑스어로 번역하세요: 나는 프로그래밍을 좋아해요.'),
        ('ai', 'J\'adore programmer.'),
        ('human', '뭐라고 말했죠?')
    ]
})

print(response.content)
