# 공통 인터페이스 예시

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-5-nano')

completion = model.invoke('반가워요!')
print(completion)

completions = model.batch(['반가워요!', '잘 있어요!'])
print(completions)

for token in model.stream('잘 있어요!'):
    print(token)


