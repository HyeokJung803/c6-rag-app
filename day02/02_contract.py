from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()

PROMPT = "당신은 신입개발자를 위한 면접관 입니다. 문장을 읽고 핵심 질문을 하시오."

# 스프링에서 DTO를 만들었듯이 class를 만듦.
# pydantic BaseModel을 상속 받음.
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    # 받은 데이터는 JSON 파싱할 필요 없이 사용
    # 데이터를 조작하고 리턴
    message = request.message
    response = client.responses.create(
        model = "gpt-5.6-luna",
        instructions=PROMPT,
        input=message
    )
    return {
        "answer" : f"입력한 메세지는 {message} \n답변 메세지 : {response.output_text}."
    }