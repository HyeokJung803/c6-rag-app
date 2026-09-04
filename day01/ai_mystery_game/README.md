# AI 추리게임 — 백야관의 11시 17분

장소 이미지, 조사 결과, 단서 보드를 갖춘 Streamlit + LangGraph 추리게임이다.

## 실행

```bash
export OPENAI_API_KEY="your-api-key"
.venv/bin/streamlit run day01/ai_mystery_game/app_mystery_game.py
```

`OPENAI_MODEL` 환경변수로 모델을 바꿀 수 있다. 왼쪽 메뉴에서 장소를 선택하면 장소 이미지가 바뀌고, 조사할수록 단서 보드가 누적된다.
