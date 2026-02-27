# 사용자 개입 패턴: Interrupt 패턴의 수동 중단 설정
import asyncio
from contextlib import aclosing

import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받습니다.'''
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={'name': tool.name}) for tool in tools],
    embeddings,
).as_retriever()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state['selected_tools']]
    res = model.bind_tools(selected_tools).invoke(state['messages'])
    return {'messages': res}


def select_tools(state: State) -> State:
    query = state['messages'][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {'selected_tools': [doc.metadata['name'] for doc in tool_docs]}


async def main():
    # 6.4절 아키텍처 이용
    builder = StateGraph(State)
    builder.add_node('select_tools', select_tools)
    builder.add_node('model', model_node)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'select_tools')
    builder.add_edge('select_tools', 'model')
    builder.add_conditional_edges('model', tools_condition)
    builder.add_edge('tools', 'model')

    graph = builder.compile(checkpointer=MemorySaver())

    input = {
        'messages': [
            HumanMessage(
                '미국 제30대 대통령의 사망 당시 나이는 몇 살이었나요?'
            )
        ]
    }

    config = {'configurable': {'thread_id': '1'}}

    # 중단 태스크 생성
    event = asyncio.Event()
    interrupt_task = asyncio.create_task(interrupt(event))

    async with aclosing(graph.astream(input, config)) as stream:
        async for chunk in stream:
            if event.is_set():
                break
            else:
                print(chunk)  # 출력

    await interrupt_task


async def interrupt(event):
    # 2초 후 중단
    await asyncio.sleep(2)
    event.set()
    print("중단 신호를 보냈습니다.")


if __name__ == '__main__':
    asyncio.run(main())
