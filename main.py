from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from ai.super_agent import SuperAgent
import asyncio
from langchain.document_loaders import TextLoader
import os

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

class Message(BaseModel):
    content: str

class DocumentLoad(BaseModel):
    file_path: str

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
        response = await super_agent.process_message(query.text)
        return AgentResponse(response=response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(message: Message):
    try:
        response = await super_agent.process_message(message.content)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-documents")
async def load_documents(doc_load: DocumentLoad):
    try:
        if not os.path.exists(doc_load.file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
            
        loader = TextLoader(doc_load.file_path)
        documents = loader.load()
        super_agent.initialize_rag(documents)
        return {"message": "문서가 성공적으로 로드되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/load")
async def load_all_documents():
    """모든 문서를 로드합니다."""
    try:
        result = await super_agent.load_all_documents()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add")
async def add_document(file_path: str):
    """새로운 문서를 추가합니다."""
    try:
        result = await super_agent.add_document(file_path)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI 테스트용
async def test_super_agent():
    test_queries = [
        "sample.txt 문서에서 인공지능 관련 내용을 찾아줘",
        "데이터베이스에서 사용자 정보를 조회해줘",
        "이 프로젝트의 주요 기능을 분석해줘"
    ]
    
    # 테스트 문서 로드
    if os.path.exists("sample.txt"):
        loader = TextLoader("sample.txt")
        documents = loader.load()
        super_agent.initialize_rag(documents)
    
    for query in test_queries:
        try:
            print(f"\n질문: {query}")
            result = await super_agent.process_message(query)
            print(f"응답: {result}")
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_super_agent()) 