from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()

PROMPT = " 당신은 신입개발자를 위한 면접관 입니다. 문장을 읽고 핵심 질문을 하세요"
class ChatRequest(BaseModel):
    message: str


@app.post("/chat", response_class=EventSourceResponse)
def chat(request: ChatRequest):
    message = request.message
    with client.responses.stream(
        model="gpt-5.6-luna",
        instructions=PROMPT,
        input=message
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                yield ServerSentEvent(event="delta", data={"delta" : delta})

# 스트리밍 예시 - JSON을 응답하는 대신, 다른 응답 객체를 반환
import time

@app.get("/demo-stream", response_class=EventSourceResponse)
def demo_stream():
    words=["FastAPI는", "응답을", "조각내서", "보낼", "수", "있어요"]
    for word in words:
        time.sleep(0.5) # 오픈AI API에서 요청후 받는 걸리는 시간
        yield ServerSentEvent(
            event="delta",
            data={"delta": word + " "}
        )
    yield ServerSentEvent(
        event="done",
        data={"done": True}
    )