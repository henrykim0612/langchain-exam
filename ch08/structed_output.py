from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# 필드에 설명을 추가해 LLM이 어느 필드에 배분할지를 결정하는 정보로 활용한다.
class Joke(BaseModel):
    setup: str = Field(description="농담의 설정")
    punchline: str = Field(description="농담의 포인트")


model = ChatOpenAI(model='gpt-5-mini', temperature=0).with_structured_output(Joke)

result = model.invoke("고양이에 대한 농담을 만들어 주세요.")
print(result)
