"""백야관의 11시 17분: 장소·단서 화면을 포함한 LangGraph 추리게임."""
import os, uuid, operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

HERE = Path(__file__).parent
MODEL = os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini")
SYSTEM = """너는 《백야관의 11시 17분》의 공정한 한국어 추리게임 GM이다.
범인은 알고 있지만 플레이어가 정당한 단서에 도달하기 전에는 범인·동기·방법을 공개하지 않는다.
관찰 사실, 진술, 해석을 구분하고, 공개되지 않은 단서나 인물을 사후에 만들지 않는다.
인물 관계는 사건 설정을 절대 기준으로 따른다. 한서린은 윤태오의 양녀이며 부인이나 배우자가 아니다.
응답은 [현재 상태] [장면과 결과] [추리 기록] [다음 행동] 순서로 작성한다.
"""
INTRO = """폭설로 외부와 고립된 산중 별장 백야관에서 재단 이사장 윤태오가 서재에서 숨진 채 발견됐습니다.
발견 시각은 밤 11시 17분. 별장 안에는 여섯 명의 용의자가 있었고, 정전과 통신 두절 때문에 누구도 쉽게 외부에 도움을 요청할 수 없었습니다.

용의자는 한서린(피해자의 양녀·재단 운영자), 강민재(법무 이사), 오지훈(보안 책임자), 문하경(주치의), 백준호(별장 관리인), 차유라(탐사보도 기자)입니다.

현장을 조사하고 용의자를 심문하며 진술, 물증, 시간대를 맞춰 범인과 사건의 전말을 밝혀내세요."""
CASE = """피해자 윤태오(58)는 폭설로 고립된 산중 별장 백야관 서재에서 23:17경 발견됐다.
용의자는 한서린(윤태오의 양녀·재단 운영자), 강민재(법무 이사), 오지훈(보안 책임자), 문하경(주치의), 백준호(관리인), 차유라(탐사보도 기자)다.
한서린은 윤태오의 배우자가 아니다.
비공개 진상: 범인 강민재, 공범 없음. 사망은 22:48~22:58 약물성 급성 심장성 쇼크.
핵심 검증 경로는 음료·약물, 전기시계와 정전 기록, 문 아래 긁힌 자국, 통화기록 지연, 장작 바구니다.
레드 헤링은 상속 갈등·취재 녹음기·처방전 분실·과거 징계다."""

LOCATIONS = {
    "서재": ("서재_현장.png", "피해자가 발견된 방. 문, 창문, 책상, 시계, 음료 잔을 조사할 수 있다."),
    "현관": ("백야관_외관.png", "폭설과 통신 두절의 흔적, 출입 기록과 눈 위 발자국을 확인할 수 있다."),
    "거실": ("백야관_외관.png", "용의자들이 모여 있던 공간. 진술과 시각을 대조할 수 있다."),
    "발전실": ("백야관_외관.png", "정전·전기시계·통신 장비 기록을 확인할 수 있다."),
}

class Route(BaseModel):
    route: Literal["현장 조사", "심문", "기록 조회", "동선 재구성", "가설 검증", "용의자 비교", "최종 추리", "일반 대화"]
    target: str

class State(TypedDict, total=False):
    question: str; route: str; target: str; turn: int
    facts: list[str]; clues: list[str]; timeline: list[str]
    log: Annotated[list[str], operator.add]; answer: str

@st.cache_resource
def build_game():
    model = init_chat_model(MODEL)
    def router(s: State):
        r = model.with_structured_output(Route).invoke([SystemMessage(content="입력을 라우트와 조사 대상으로 분류하라."), HumanMessage(content=s["question"])])
        return {"route": r.route, "target": r.target, "log": [f"router → {r.route}"]}
    def supervisor(s: State):
        return {"turn": s.get("turn", 0) + 1, "log": [f"supervisor → {s['target']}"]}
    def investigate(s: State):
        known = "\n".join(s.get("facts", [])[-10:]) or "없음"
        prompt = f"{SYSTEM}\n사건 설계:\n{CASE}\n이미 공개된 기록:\n{known}\n플레이어 입력: {s['question']}\n라우트: {s['route']} / 대상: {s['target']}\n이번 턴 조사 결과를 구체적으로 작성하라. 새 핵심 단서는 최대 2개만 공개하라. 최종 진상은 말하지 마라. 인물 관계를 임의로 바꾸거나 추가하지 마라."
        answer = model.invoke(prompt).content
        facts = [*s.get("facts", []), answer]
        clues = [*s.get("clues", []), answer]
        return {"answer": answer, "facts": facts, "clues": clues, "log": ["investigate → 결과 저장"]}
    g = StateGraph(State)
    for name, fn in [("router", router), ("supervisor", supervisor), ("investigate", investigate)]: g.add_node(name, fn)
    g.add_edge(START, "router"); g.add_edge("router", "supervisor"); g.add_edge("supervisor", "investigate"); g.add_edge("investigate", END)
    return g.compile(checkpointer=InMemorySaver())

st.set_page_config(page_title="백야관의 11시 17분", page_icon="🔎", layout="wide")
st.title("🔎 백야관의 11시 17분")
st.caption("폭설로 고립된 별장. 당신은 유일한 탐정입니다.")
with st.container(border=True):
    st.subheader("사건 개요")
    st.markdown(INTRO)
if "thread_id" not in st.session_state: st.session_state.thread_id = f"case-{uuid.uuid4().hex[:8]}"
if "game" not in st.session_state: st.session_state.game = build_game()
if "history" not in st.session_state: st.session_state.history = []
if "facts" not in st.session_state: st.session_state.facts = []

with st.sidebar:
    st.header("조사 메뉴")
    place = st.selectbox("장소", list(LOCATIONS))
    image_name, description = LOCATIONS[place]
    st.caption(description)
    st.divider()
    st.subheader("단서 보드")
    if st.session_state.facts:
        for i, fact in enumerate(st.session_state.facts, 1): st.markdown(f"**{i}.** {fact[:180]}")
    else: st.caption("아직 확보한 단서가 없습니다.")
    if st.button("새 사건 시작", use_container_width=True):
        st.session_state.thread_id = f"case-{uuid.uuid4().hex[:8]}"; st.session_state.history = []; st.session_state.facts = []; st.rerun()

# 사진 영역은 왼쪽에 고정하고, 오른쪽 대화 영역만 갱신한다.
photo_col, chat_col = st.columns([1.15, 1], gap="large")
with photo_col:
    st.image(str(HERE / "assets" / image_name), caption=f"{place} 현장", use_container_width=True)
    st.subheader(f"📍 {place}")
    st.write(description)
    st.info("사진을 살펴본 뒤, 아래 입력창에 조사할 대상을 구체적으로 입력하세요.")

with chat_col:
    st.subheader("💬 조사 기록")
    chat_history = st.container(height=620, border=True)
    with chat_history:
        if not st.session_state.history:
            st.info("첫 조사: 아래 입력창에 조사 행동을 입력하세요. 예: 서재의 문과 시계를 현장 조사해")
        for item in st.session_state.history:
            with st.chat_message(item["role"]): st.markdown(item["content"])

    q = st.chat_input("조사·심문·기록 조회·추리를 입력하세요")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    out = st.session_state.game.invoke({"question": q, "facts": st.session_state.facts, "log": []}, {"configurable": {"thread_id": st.session_state.thread_id}})
    st.session_state.facts = out.get("facts", st.session_state.facts)
    st.session_state.history.append({"role": "assistant", "content": out["answer"]})
    st.rerun()
