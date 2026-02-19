# 상태 그래프 생성(체크포인터 추가)

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

model = ChatOpenAI(model='gpt-5-mini')


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

# 그래프에 체크포인터 추가
# 각 단계가 종료될 때마다 상태가 기록되므로, 최초 실행 이후의 모든 호출은 백지 상태로 시작하지 않는다.
# 그래프 호출 시, 체크포인터를 활용해 저장된 최신 상태(존재할 경우)를 불러온 후, 새 입력값과 결합한다.
graph = builder.compile(checkpointer=MemorySaver())

# 스레드 설정
thread1 = {'configurable': {'thread_id': '1'}}

# 영속성 추가 후 그래프 실행
result_1 = graph.invoke({'messages': [HumanMessage('안녕하세요, 저는 민혁입니다!')]}, thread1)
result_2 = graph.invoke({'messages': [HumanMessage('제 이름이 뭐죠?')]}, thread1)

# 상태 업데이트도 가능함
# 상태가 저장하고 있는 메시지 목록에 새 메시지가 추가되며, 동일 스레드에서의 그래프를 호출할 때 추가된 메시지가 활용된다.
graph.update_state(thread1, {'messages': [HumanMessage('저는 LLM이 좋아요!')]})

# 상태 확인
print(graph.get_state(thread1))
