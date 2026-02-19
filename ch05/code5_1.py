# 상태 그래프 구성

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    # 메시지의 유형은 list이다.
    # 어노테이션의 `add_messages` 함수는 상태를 업데이트하는 방법이다.
    # 이 경우 이전 메시지를 대체하는 대신 새 메시지를 추가한다.
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOpenAI(model='gpt-5-mini')


def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {'messages': [answer]}


# 챗봇 노드 추가
# 첫 번째 인자는 고유한 노드 이름
# 두 번째 인자는 실행할 함수 또는 Runnable
builder.add_node('chatbot', chatbot)

# 엣지 추가
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

# 그래프 이미지 저장
graph.get_graph().draw_mermaid_png(output_file_path="LLM_call_architecture.png")

# 그래프 실행
input = {'messages': [HumanMessage('안녕하세요!')]}
for chunk in graph.stream(input):
    print(chunk)
