# 기본 모델 호출
from langchain_openai.llms import OpenAI

model = OpenAI(model='gpt-3.5-turbo-instruct')

print(model.invoke('하늘이'))
