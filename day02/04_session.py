from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()

PROMPT = " 당신은 신입개발자를 위한 면접관 입니다. 문장을 읽고 핵심 질문을 하세요"

# [{"role":"user","content":"..."}, ..., ..., ]
HISTORY: list[dict] = [] # 일반 히스토리 저장소
# {"A" : [{"role":"user","content":"..."}, ..., ..., ],
#  "B" : [{"role":"user","content":"..."}, ..., ..., ], ... }
SESSIONS: dict[str, list[dict]] = {} # 세션 저장소

class ChatRequest(BaseModel):
    session_id: str # 각 대화를 구분하기 위한 식별자
    message: str

# 응답이 SSE 이벤트 소스임을 지정
@app.post("/chat", response_class=EventSourceResponse)
def chat(request: ChatRequest):
    # 세션 저장소 초기화하고, 히스토리에 유저 메시지 저장
    history = SESSIONS.setdefault(request.session_id, [])
    history.append({ "role":"user", "content": request.message})
    # HISTORY.append({ "role":"user", "content": request.message})
    answer_parts: list[str] = []

    # 스트리밍 API로 받음
    with client.responses.stream(
        model="gpt-5.6-luna",
        instructions=PROMPT,
        input=history   # 세션의 주고받은 것들을 요청 context
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                answer_parts.append(delta)  # 응답 받은 토큰들을 연결
                yield ServerSentEvent(event="delta", data={"delta": delta})

    # 응답 내역 추가
    full_answer = "".join(answer_parts)
    history.append({"role":"assistant", "content": full_answer})
    yield ServerSentEvent(event="done", data={"done": True})