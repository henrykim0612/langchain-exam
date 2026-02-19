# 채팅 기록 수정: 메시지 축약 방식

from langchain_core.messages import (HumanMessage, SystemMessage, AIMessage, trim_messages)
from langchain_openai import ChatOpenAI

# 샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.'),
    HumanMessage(content='안녕하세요! 나는 민혁입니다.'),
    AIMessage(content='안녕하세요!'),
    HumanMessage(content='바닐라 아이스크림을 좋아해요.'),
    AIMessage(content='좋네요!'),
    HumanMessage(content='2 + 2는 얼마죠?'),
    AIMessage(content='4입니다.'),
    HumanMessage(content='고마워요.'),
    AIMessage(content='천만에요!'),
    HumanMessage(content='즐거운가요?'),
    AIMessage(content='예!'),
]

# 축약 설정
trimmer = trim_messages(
    max_tokens=65,
    strategy='last',  # 메시지 목록을 뒤부터 max_tokens 개의 토큰을 받는다
    token_counter=ChatOpenAI(model='gpt-5-mini'),
    include_system=True,
    allow_partial=False,  # 제한 범위 내에 마지막 메시지의 내용을 포함시키기 위해 해당 메시지의 일부를 생략할지 결정. False 이므로 총합이 한도를 초과하면 메시지를 완전히 제거.
    start_on='human',  # 잘라낸 후 어떤 메시지 타입부터 시작하도록 강제할 것인지.
)

# 축약 적용
trimmed = trimmer.invoke(messages)
print(trimmed)
