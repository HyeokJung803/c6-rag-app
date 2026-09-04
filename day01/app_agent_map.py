"""📋 이 파일은 통째로 붙여넣어 실행하는 앱이다(직접 타이핑 대상 아님).

Agent Map — 라우팅·재검색·관찰이 한 화면에 보이는 지식 도우미.

실행:  streamlit run app_agent_map.py
질문을 넣으면 그래프가 어느 길로 갔는지(경로), 왜 멈췄는지(stop_reason),
그리고 이 실행의 LangSmith trace 링크를 함께 보여 준다.
그래프 '구조' 그림은 미리 만들어 둔 graph.png 를 쓴다(오프라인 대비).

읽는 법: 아래 그래프(router·retrieve·rewrite·grade·generate·giveup·web·calc·direct)는
**02_routing_repair.py 에서 만든 것과 같다**(화면용으로 짧게 압축). 각 노드의 뜻은 02를 참고.
이 파일에서 새로 볼 것은 세 가지다 — ① Streamlit 화면, ② @st.cache_resource(그래프를 한 번만 만들기),
③ 질문마다 trace 링크를 집어 화면에 띄우기.
"""
import os, operator, ast, operator as _op
import uuid   # 질문마다 겹치지 않는 thread_id를 만든다(질문끼리 상태가 섞이지 않게 — 02와 같은 이유)
from pathlib import Path
from typing import Literal, TypedDict, Annotated

import streamlit as st
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.run_helpers import get_current_run_tree
from tavily import TavilyClient

os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "agent-map")

HERE = Path(__file__).parent
MAX_ATTEMPTS = 2

DOCS = [
    "연차 유급휴가는 1년간 80퍼센트 이상 출근한 근로자에게 15일이 주어진다.",
    "과정 수료 기준은 출석률 80퍼센트 이상이다.",
    "취업지원 프로그램 신청은 매 학기 초 2주 안에 해야 한다.",
    "수업에서 만든 코드는 개인 저장소에 백업하는 것을 권장한다.",
    "점심 시간은 12시부터 1시까지이며 강의실 취식은 금지된다.",
]

_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.USub: _op.neg, ast.Pow: _op.pow}


def safe_calc(expr: str):
    def _ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):   return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
        if isinstance(n, ast.UnaryOp): return _OPS[type(n.op)](_ev(n.operand))
        raise ValueError("허용되지 않은 식")
    return _ev(ast.parse(expr, mode="eval").body)


class Route(BaseModel):
    datasource: Literal["docs", "web", "calc", "direct"] = Field(
        description="docs=사내 규정·휴가·수료·시설, web=최신 외부 정보, calc=수식 계산, direct=상식·잡담")


class S(TypedDict):
    question: str
    route: str
    query: str
    hits: list
    attempts: int
    stop_reason: str
    answer: str
    trace_url: str
    log: Annotated[list, operator.add]


# @st.cache_resource: Streamlit은 버튼을 누를 때마다 이 파일 전체를 처음부터 다시 실행한다.
# 이 표시가 붙은 함수의 결과(그래프·모델·checkpointer)는 캐시되어, 재실행 때 새로 만들지 않고 재사용된다.
@st.cache_resource
def build_app():
    """그래프는 한 번만 만들어 재사용한다(질문마다 새로 만들지 않는다)."""
    fast = init_chat_model("openai:gpt-5.4-nano")   # 분류·재작성(가벼움)
    main = init_chat_model("openai:gpt-5.6-luna")   # 답변(기본)
    tavily = TavilyClient(os.environ["TAVILY_API_KEY"])

    def _url():   # 현재 실행의 trace 링크를 집는다(04에서 배운 방식). 트레이싱이 꺼져 있으면 빈 문자열.
        rt = get_current_run_tree()
        return rt.get_url() if rt is not None else ""

    def router(s: S):
        r = fast.with_structured_output(Route).invoke(
            "질문을 docs/web/calc/direct 중 하나로 분류하라. 회사·과정의 규정/휴가/수료/시설 질문은 docs 다.\n"
            f"질문: {s['question']}")
        return {"route": r.datasource, "query": s["question"], "attempts": 0,
                "log": [f"router → {r.datasource}"]}

    def retrieve(s: S):
        toks = [t for t in s["query"].split() if len(t) >= 2]
        hits = [d for d in DOCS if any(t in d for t in toks)]
        return {"hits": hits, "attempts": s["attempts"] + 1,
                "log": [f"retrieve#{s['attempts']+1} (hits={len(hits)})"]}

    def rewrite(s: S):
        q = fast.invoke(f"검색 실패. 핵심 명사 1~2개만 공백으로: {s['question']}").content.strip()
        return {"query": q, "log": [f"rewrite → {q!r}"]}

    def grade(s: S) -> Literal["generate", "rewrite", "giveup"]:
        if s["hits"]:                      return "generate"
        if s["attempts"] >= MAX_ATTEMPTS:  return "giveup"
        return "rewrite"

    def generate(s: S):
        a = main.invoke(f"근거:\n{chr(10).join(s['hits'])}\n질문: {s['question']}\n근거로만 짧게 답하라.").content
        return {"answer": a, "stop_reason": "answered", "trace_url": _url(), "log": ["generate"]}

    def giveup(s: S):
        return {"answer": "관련 근거를 찾지 못해 답하지 않습니다.",
                "stop_reason": "max_attempts", "trace_url": _url(), "log": ["giveup"]}

    def web(s: S):
        r = tavily.search(s["question"], max_results=3)
        found = "\n".join(f"- {x['title']}: {x['url']}" for x in r["results"])
        a = main.invoke(f"웹 결과:\n{found}\n질문: {s['question']}\n짧게 답하라.").content
        return {"answer": a, "hits": [found], "stop_reason": "web", "trace_url": _url(), "log": ["web(Tavily)"]}

    def calc(s: S):
        expr = fast.invoke(   # 위에서 만든 fast 모델을 재사용(매번 새로 만들지 않는다)
            f"계산할 수식만 파이썬 문법으로(설명 금지): {s['question']}").content.strip()
        try:
            a, reason = f"{expr} = {safe_calc(expr)}", "calc"
        except Exception:
            a, reason = "수식을 계산할 수 없습니다.", "calc_fail"
        return {"answer": a, "stop_reason": reason, "trace_url": _url(), "log": [f"calc {expr!r}"]}

    def direct(s: S):
        return {"answer": main.invoke(s["question"]).content,
                "stop_reason": "direct", "trace_url": _url(), "log": ["direct"]}

    g = StateGraph(S)
    for name, fn in [("router", router), ("retrieve", retrieve), ("rewrite", rewrite),
                     ("generate", generate), ("giveup", giveup),
                     ("web", web), ("calc", calc), ("direct", direct)]:
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"docs": "retrieve", "web": "web", "calc": "calc", "direct": "direct"})
    g.add_conditional_edges("retrieve", grade,
                            {"generate": "generate", "rewrite": "rewrite", "giveup": "giveup"})
    g.add_edge("rewrite", "retrieve")
    for n in ["generate", "giveup", "web", "calc", "direct"]:
        g.add_edge(n, END)
    return g.compile(checkpointer=InMemorySaver())


def run_once(question: str):
    app = build_app()   # 캐시된 그래프를 가져온다(처음 한 번만 실제로 만든다)
    # 질문마다 새 thread_id를 준다. 같은 id를 재사용하면 checkpointer가 이전 질문의 로그까지
    # 이어 붙여 '실제 지나간 경로'가 뒤섞인다(02에서 배운 그대로). 그래서 매번 독립 대화로 돌린다.
    thread_id = f"q-{uuid.uuid4().hex[:8]}"
    return app.invoke({"question": question, "log": []},
                      {"configurable": {"thread_id": thread_id}, "recursion_limit": 25})


# ---------------- 화면 ----------------
st.set_page_config(page_title="Agent Map", page_icon="🗺️", layout="wide")
st.title("🗺️ Agent Map — 라우팅 지식 도우미")
st.caption("질문이 그래프의 어느 길로 갔고, 왜 멈췄고, 실행 기록(trace)은 어디 있는지 함께 봅니다.")

left, right = st.columns([1, 1])

with left:
    st.subheader("그래프 구조")
    graph_png = HERE / "graph.png"
    if graph_png.exists():
        st.image(str(graph_png), use_container_width=True)
    else:
        st.info("graph.png 가 없습니다. 02_routing_repair.py 를 한 번 실행해 만드세요.")

with right:
    st.subheader("질문하기")
    q = st.text_input("질문", value="연차 유급휴가는 며칠인가요?")
    if st.button("실행", type="primary") and q.strip():
        with st.spinner("그래프 실행 중…"):
            out = run_once(q)   # 질문 한 건을 그래프에 태우고, 최종 state를 받는다
        # state에 다 드러내 둔 값을 그대로 화면 지표로 보여 준다(그래서 state 설계가 곧 화면이다).
        st.metric("경로(route)", out.get("route", "-"))          # 라우터가 고른 갈래
        c1, c2 = st.columns(2)
        c1.metric("시도(attempts)", out.get("attempts", 0))       # 문서 검색 시도 횟수
        c2.metric("멈춘 이유(stop)", out.get("stop_reason", "-"))  # 왜 끝났나
        st.markdown("**실제 지나간 경로**")
        st.code(" → ".join(out.get("log", [])) or "(로그 없음)")   # 노드들이 남긴 로그 = 지나온 길
        st.markdown("**답변**")
        st.write(out.get("answer", ""))
        if out.get("trace_url"):
            st.markdown(f"🔗 [이 실행의 LangSmith trace 열기]({out['trace_url']})")