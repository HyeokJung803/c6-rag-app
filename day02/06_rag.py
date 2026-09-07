# 6단계 — 내 RAG가 서비스가 된다: 백엔드 함수 속만 갈아끼운다
#
# 실행:  fastapi dev 06_rag.py
#
# 5단계(완성 골격)에서 계약·SSE·session·CORS·화면은 '그대로'. 딱 하나가 바뀐다:
#   면접 코치가 답하기 전에, 앞 과정에서 배운 방식 그대로 **문서를 임베딩해 벡터 검색**으로 근거를 찾는다.
#   여기 검색기는 앞 과정에서 쓰던 것과 똑같다 — Chroma(벡터 DB) + OpenAI 임베딩. 새로 만들지 않고 가져와 꽂는다.
#   그래서 두루뭉술한 면접이 아니라 '이 공고의 구체 요건'에 근거해 묻고, 근거에 없으면 모른다고 한다.
#   이것이 "내 함수를 서비스로"의 완성이다: 서비스 껍데기는 그대로, 함수 속만 진짜 RAG로.
#
# 왜 FastAPI 였나가 여기서 드러난다: 검색 로직이 전부 Python 이라, 별도 서비스로 떼거나 포팅하지 않고
# /chat 함수 속에 그대로 꽂는다. RAG·모델 호출과 '같은 언어'라는 것이 오늘 이 프레임워크를 고른 진짜 이유다.
#   (지금 Chroma는 이 프로세스 안에서만 산다. 팀이 함께 쓰고 서버가 꺼져도 남는 '영구 벡터 DB'는 뒤 수업에서.)

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings   # 앞 과정에서 쓰던 그 임베딩(문장 → 의미 벡터)
from langchain_chroma import Chroma             # 앞 과정에서 쓰던 그 벡터 DB(벡터를 담고 가까운 것을 찾아 줌)

load_dotenv()
client = OpenAI()
MODEL = "gpt-5.6-luna"
EMBED_MODEL = "text-embedding-3-large"          # 앞 과정에서 쓰던 임베딩 계열(짧은 공고엔 큰 쪽이 더 안정적)

# ── 이 회사만의 채용 공고 ── 모델이 근거 없이는 알 수 없는 구체값(급여는 일부러 넣지 않았다 — '모른다' 시연용)
JOB_POSTING = [
    "채용 직무는 백엔드 개발 신입이며, 주로 Python과 FastAPI로 API 서버를 만든다.",
    "지원 자격은 자료구조와 데이터베이스 기초를 이해하고, 개인 프로젝트를 1건 이상 만들어 본 경험이다.",
    "우대 사항은 클라우드(AWS 등) 배포 경험과 오픈소스 기여 이력이다.",
    "근무 형태는 주 3일 사무실 출근, 2일 재택근무의 혼합 근무다.",
    "전형 절차는 서류 검토, 1차 실무 코딩 면접, 2차 인성 면접 순이며 총 2주 안에 결과를 통보한다.",
    "지원은 회사 채용 페이지에서 이력서와 포트폴리오 링크를 제출하면 된다.",
    "수습 기간은 3개월이며, 수습 중에도 정규직과 동일한 복지를 제공한다.",
    "복지로 도서 구입비와 컨퍼런스 참가비를 연 100만 원까지 지원한다.",
]

# 공고 문장들을 한 번 임베딩해 벡터 DB(Chroma)에 넣는다. 앞 과정에서 배운 Chroma.from_texts 그대로다.
# 서버가 뜰 때 딱 한 번 실행된다(요청마다가 아니라). 이 STORE 가 곧 '검색 가능한 지식'이 된다.
STORE = Chroma.from_texts(JOB_POSTING, OpenAIEmbeddings(model=EMBED_MODEL))

INTERVIEWER = ("너는 아래 '채용 공고 근거'를 바탕으로 진행하는 모의 면접 코치다. "
               "지원자가 직무·자격·절차를 물으면 근거에 있는 내용으로만 한국어로 짧게 답한다. "
               "그 밖에는 공고의 구체 요건에 비추어 꼬리 질문을 딱 하나 던진다. "
               "근거에 답이 없으면 지어내지 말고 '공고에 없어 모른다'고 답한다. 전체 2~3문장 이내.")


def retrieve(query: str, k: int = 3) -> list[str]:
    # 뜻으로 찾는다(RAG의 R = Retrieval): 질문을 임베딩해 공고 벡터와 '가까운' 상위 k개를 가져온다.
    # 글자가 안 겹쳐도(예: 질문 "연봉" ↔ 문서 "급여") 의미가 가까우면 찾는다 — 그게 임베딩 검색의 핵심.
    hits = STORE.similarity_search(query, k=k)
    return [d.page_content for d in hits]        # Document 객체에서 본문 문자열만 뽑아 돌려준다


class ChatIn(BaseModel):
    session_id: str
    message: str


SESSIONS: dict[str, list] = {}

app = FastAPI(title="AI 모의 면접 코치 (RAG)")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "mode": "rag"}


@app.post("/chat", response_class=EventSourceResponse)
def chat(inp: ChatIn):
    history = SESSIONS.setdefault(inp.session_id, [])    # 이 세션의 대화 이력 — '질문/답'만 쌓는다
    hits = retrieve(inp.message)                         # ← 새로 생긴 한 걸음: 벡터 검색으로 이번 질문의 근거를 찾는다
    context = "\n".join(hits) if hits else "(관련 공고 항목 없음)"
    yield ServerSentEvent(data={"sources": hits}, event="sources")   # 어떤 근거를 썼는지 먼저 화면에 보낸다(투명성)

    # ── turn-local(이번 턴에만 근거를 끼운다) ── 이 분리가 이 파일에서 가장 중요한 판단이다.
    # 이력(history)에는 '원문 질문'만 남기고, 검색 근거는 '이번에 모델로 보내는 판'에만 붙인다.
    # 그렇게 하지 않고 근거까지 이력에 쌓으면: ① 과거 턴의 공고 문서가 미래 질문에 계속 딸려 들어가 토큰이 불고,
    #   ② 지난 근거와 이번 근거가 뒤섞여 모델이 엉뚱한 근거로 답할 수 있다.
    history.append({"role": "user", "content": inp.message})            # 이력엔 원문 질문만
    grounded = f"[채용 공고 근거]\n{context}\n\n[지원자] {inp.message}"   # 근거+질문을 합친 '이번 턴용' 메시지
    model_input = history[:-1] + [{"role": "user", "content": grounded}]  # 마지막 원문 질문만 근거 붙인 판으로 교체

    answer = ""
    with client.responses.stream(model=MODEL, instructions=INTERVIEWER, input=model_input) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                answer += event.delta
                yield ServerSentEvent(data={"delta": event.delta}, event="delta")
    history.append({"role": "assistant", "content": answer})           # 답은 이력에 그대로 저장(다음 턴이 이어짐)
    yield ServerSentEvent(data={"turns": len(history)}, event="done")


app.frontend("/", directory="./day02/static")
