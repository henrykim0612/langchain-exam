# 명령형 구성을 사용한 스트리밍 호출 예시

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# 구성 요소
template = ChatPromptTemplate.from_messages([
    ('system', '당신은 친절한 어시스턴트입니다.'),
    ('human', '{question}')
])

model = ChatOpenAI(model='gpt-5-mini')


@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


for part in chatbot.stream({
    'question': '거대 언어 모델은 어디서 제공하나요?!'
}):
    print(part)
