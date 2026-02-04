# 명령형 구성을 사용한 스트리밍 호출 예시(비동기 방식)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# 구성 요소
template = ChatPromptTemplate.from_messages([
    ('system', '당신은 친절한 어시스턴트입니다.'),
    ('human', '{question}')
])

model = ChatOpenAI(model='gpt-5-nano')

@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

async def main():
    return await chatbot.ainvoke({'question': '거대 언어 모델은 어디서 제공하나요?'})

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))