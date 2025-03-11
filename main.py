from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from ai.super_agent import SuperAgent

import models
from database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
super_agent = SuperAgent()

# 요청 모델 정의
class Query(BaseModel):
    text: str

# 응답 모델 정의
class AgentResponse(BaseModel):
    response: str

@app.get("/users/")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return user

@app.post("/ask/", response_model=AgentResponse)
async def ask_agent(query: Query):
    try:
        # 슈퍼에이전트에게 질문하고 응답 받기
        response = super_agent.run(query.text)
        return AgentResponse(response=response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"} 