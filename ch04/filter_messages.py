# 채팅 기록 수정: 메시지 필터링

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, filter_messages)
from langchain_openai import ChatOpenAI

# 샘플 메시지
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.', id='1'),
    HumanMessage(content='예시 입력', id='2', name='example_user'),
    AIMessage(content='예시 출력', id='3', name='example_assistant'),
    HumanMessage(content='실제 입력', id='4', name='bob'),
    AIMessage(content='실제 출력', id='5', name='alice'),
]

# 사용자 메시지만 필터링
human_messages = filter_messages(messages, include_types='human')
print('사용자 메시지:', human_messages)

# 특정 이름의 메시지 제외
excluded_names = filter_messages(
    messages, exclude_names=['example_user', 'example_assistant']
)
print('\n이름에 example이 포함되지 않은 메시지:', excluded_names)

# 유형과 ID로 필터링
filtered_messages = filter_messages(
    messages, include_types=['human', 'ai'], exclude_ids=['3']
)
print('\n특정 유형과 ID로 필터링한 메시지:', filtered_messages)

# 선언형 구성
model = ChatOpenAI(model='gpt-5-mini')
filter_ = filter_messages(exclude_names=['example_user', 'example_assistant'])
chain = filter_ | model
