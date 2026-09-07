# Room, in words

말로 시간·날씨·조명·소품을 바꾸는 2D 방입니다. `demo.py`는 FastAPI와 상태 저장을,
`index.html`은 SVG 방과 화면 조작을 담당합니다.

프로젝트 루트에서 실행합니다. 기존 가상환경의 패키지를 사용합니다.

```bash
.venv/bin/fastapi dev day02/demo/demo.py
```

브라우저에서 <http://127.0.0.1:8000>을 엽니다. API 문서는 `/docs`입니다.
API 키 없이도 네 가지 분위기 버튼과 되돌리기·초기화를 사용할 수 있습니다.

자유로운 문장 입력은 프로젝트 루트 또는 `day02/demo/.env`에 다음을 설정하고
서버를 재시작하면 사용할 수 있습니다. `.env`는 커밋하지 마세요.

```dotenv
OPENAI_API_KEY=본인의_API_키
OPENAI_MODEL=gpt-5.6-luna
```

모델 기본값은 기존 day02 예제와 같습니다. 계정에서 사용할 수 있는
Structured Outputs 지원 모델로 `OPENAI_MODEL`을 변경할 수 있습니다.
자유 입력은 OpenAI API를 호출하며 사용 요금이 발생합니다. 예시 버튼은 API를 호출하지 않습니다.

입력 예시:

- 비 오는 밤에 책상 조명만 켜줘
- 지금 상태에서 조명만 보라색으로 바꿔줘
- 화분과 턴테이블을 치워줘
- 아까로 되돌려줘

`POST /sessions`로 방을 생성하고, `/sessions/{session_id}/command`로 문장을 보냅니다.
GPT가 반환한 상태를 Pydantic으로 검증한 뒤 적용합니다. 오류나 응답 거절 시에는
기존 방과 되돌리기 기록을 유지합니다. 구조화 응답 방식은
[OpenAI 공식 문서](https://developers.openai.com/api/docs/guides/structured-outputs)를 참고했습니다.

방 상태는 서버 메모리에, 방 ID는 브라우저 탭의 sessionStorage에 저장합니다.
최근 30단계를 되돌릴 수 있으며 서버 재시작 시 초기화됩니다.
단일 서버 프로세스로 실행하는 로컬 실습용이며 음악 재생·음성 인식 기능은 없습니다.
