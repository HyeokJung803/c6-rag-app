from fastapi import FastAPI

app = FastAPI()

# 함수 -> API 서비스로
# web Method와 Path 경로를 앱에 등록
@app.get("/health")
def health():
    return { "status" : "ok"}

@app.get("/hello")
def health():
    return "안녕"


# 경로 변수
@app.get("/users/{user_id}")
def get_user(user_id:int):
    return {"user_id": user_id}

# 쿼리 변수
@app.get("/search")
def search_item(q: str = None, page : int = 1):
    return {"query" : q, "page" : page}
    


# 실행 uv run fastapi dev [파일명]