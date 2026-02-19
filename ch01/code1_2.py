# 채팅 모델 호출

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model='gpt-5-mini')
prompt = [HumanMessage('프랑스의 수도는 어디인가요?')]

print(model.invoke(prompt))
