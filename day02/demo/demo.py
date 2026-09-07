"""실행(프로젝트 루트): .venv/bin/fastapi dev day02/demo/demo.py"""

import os
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ConfigDict

BASE = Path(__file__).resolve().parent
load_dotenv(BASE.parents[1] / ".env")
load_dotenv(BASE / ".env")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
app = FastAPI(title="Room, in words")


class Room(BaseModel):
    time: Literal["day", "sunset", "night"] = "sunset"
    weather: Literal["clear", "rain", "snow"] = "clear"
    ceiling_light: bool = False
    desk_light: bool = True
    light_color: Literal["warm", "white", "purple", "blue", "red"] = "warm"
    plant: bool = True
    record_player: bool = True


class Decision(BaseModel):
    action: Literal["update", "undo", "reset", "keep"]
    room: Room
    message: str


class Command(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=500)


# 단일 프로세스 실습용 저장소. 서버 재시작 시 초기화된다.
SESSIONS: dict[str, dict] = {}
REGISTRY_LOCK = Lock()
PRESETS = {
    "rain": Room(time="night", weather="rain", light_color="blue"),
    "focus": Room(time="night", desk_light=True, plant=True, record_player=False),
    "snow": Room(time="day", weather="snow", ceiling_light=True, light_color="warm"),
    "late": Room(time="night", ceiling_light=True, light_color="purple"),
}
INSTRUCTIONS = """너는 2D 방의 상태를 조작한다. 현재 상태와 사용자의 말을 보고 결정하라.
time은 day/sunset/night, weather는 clear/rain/snow, 조명 두 개는 켜짐 여부,
light_color는 warm/white/purple/blue/red, plant와 record_player는 소품 표시 여부다.
언급하지 않은 값은 유지한다. 분위기를 요청하면 관련 값을 조합한다.
이전 상태로 돌아가기는 undo, 처음으로 초기화는 reset이다.
지원하지 않는 요청(음악 재생, 가구 추가, 음성 등)은 keep으로 하고 가능한 기능을 설명한다.
message는 실제 변경에 맞는 짧은 한국어 한 문장이다. 실행할 코드나 HTML을 만들지 마라.
"""


def session_for(session_id: str):
    with REGISTRY_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "방이 만료되었어요. 페이지를 새로고침해 주세요.")
    return session


def snapshot(session, message=""):
    return {"room": session["room"], "can_undo": bool(session["history"]), "message": message}


def apply_action(session, action, room=None, message=""):
    if action == "undo":
        if not session["history"]:
            return snapshot(session, "아직 되돌릴 변경이 없어요.")
        session["room"] = session["history"].pop()
        return snapshot(session, "한 단계 전의 방으로 돌아왔어요.")
    target = Room() if action == "reset" else room
    if action in ("reset", "update") and target != session["room"]:
        session["history"].append(session["room"].model_copy())
        session["history"] = session["history"][-30:]
        session["room"] = target
    return snapshot(session, message)


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "ai_ready": bool(os.getenv("OPENAI_API_KEY")), "model": MODEL}


@app.post("/sessions")
def create_session():
    with REGISTRY_LOCK:
        if len(SESSIONS) >= 1000:
            raise HTTPException(503, "실습용 방이 가득 찼어요. 서버를 재시작해 주세요.")
        session_id = str(uuid4())
        session = {"room": Room(), "history": [], "lock": Lock()}
        SESSIONS[session_id] = session
    return {"session_id": session_id, **snapshot(session)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = session_for(session_id)
    with session["lock"]:
        return snapshot(session)


@app.post("/sessions/{session_id}/actions/{action}")
def action(session_id: str, action: Literal["undo", "reset", "rain", "focus", "snow", "late"]):
    session = session_for(session_id)
    with session["lock"]:
        if action in PRESETS:
            return apply_action(session, "update", PRESETS[action].model_copy(), "선택한 분위기로 바꿨어요.")
        return apply_action(session, action, message="처음의 방으로 돌아왔어요.")


@app.post("/sessions/{session_id}/command")
def command(session_id: str, inp: Command):
    session = session_for(session_id)
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "자유 입력은 OPENAI_API_KEY 설정이 필요해요. 아래 분위기 버튼은 바로 사용할 수 있어요.")
    # 같은 방의 명령을 순서대로 처리하고, 실패하면 상태와 되돌리기 기록을 유지한다.
    with session["lock"]:
        try:
            with OpenAI(timeout=30, max_retries=0) as client:
                response = client.responses.parse(
                    model=MODEL,
                    instructions=INSTRUCTIONS,
                    input=f"현재 방: {session['room'].model_dump_json()}\n요청: {inp.message}",
                    text_format=Decision,
                )
            result = response.output_parsed
            if result is None:
                raise HTTPException(502, "변경을 해석하지 못했어요. 다른 표현으로 말해 주세요.")
        except OpenAIError as exc:
            raise HTTPException(502, "AI 연결에 실패했어요. 서버의 API 키, 모델 설정과 연결을 확인해 주세요.") from exc
        return apply_action(session, result.action, result.room, result.message)
