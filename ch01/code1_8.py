# JSON 형식 출력 요청

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class AnswerWithJustification(BaseModel):
    '''사용자의 질문에 대한 답변과 그에 대한 근거(Justification)를 함께 제공하세요.'''
    answer: str
    '''사옹자의 질문에 대한 답변'''
    justification: str
    '''답변에 대한 근거'''


llm = ChatOpenAI(model='gpt-5-mini', temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)
# with_structured_output 메서드는 다음과 같은 용도로 활용한다.
# - 스키마를 JSONSchema 객체로 변환해 LLM에 전송한다. 해당 객체는 JSON 데이터의 구조[타입, 이름, 설명]을 기술한다.
#   랭체인은 각 LLM에서 이를 수행할 최선의 방법을 선택한다. 주로 함수 호출과 프롬프트 작성에 많이 쓰인다.
# - 스키마는 LLM이 반환한 출력물을 반환하기 전에 그 유효성을 검증한다. 이를 통해 출력 겨로가가 스키마를 정확히 준수하는지 확인한다.

result = structured_llm.invoke('''1 킬로그램의 벽돌과 1 킬로그램의 깃털 중 어느 쪽이 더 무겁나요?''')

print(result.model_dump_json())
