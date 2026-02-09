# 상태 그래프 생성

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # 메시지의 유형은 list이다.
    # 어노테이션의 `add_messages` 함수는 상태를 업데이트하는 방법이다.
    # 이 경우 이전 메시지를 대체하는 대신 새 메시지를 추가한다.
    # Annotated 는 "이 타입에 추가 동작/의미를 붙인다"


builder = StateGraph(State)

model = ChatOpenAI(model='gpt-5-nano')


def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {'messages': [answer]}


# Chatbot 노드 추가
# 첫 번째 인자는 고유한 노드 이름
# 두 번째 인자는 실행할 함수 또는 Runnable
builder.add_node('chatbot', chatbot)

# 엣지 추가
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

# 그래프 이미지 저장
graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

# 그래프 실행
input = {'messages': [HumanMessage('안녕하세요!')]}
for chunk in graph.stream(input):  # stream 함수를 사용하면 그래프의 각 단계가 완료된 후 상태의 전체 값을 스트리밍한다.
    print(chunk)
