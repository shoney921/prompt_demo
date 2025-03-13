from fastapi import APIRouter, HTTPException
from ai.super_agent import SuperAgent
from ai.graph_super_agent import GraphSuperAgent

router = APIRouter()

langchain_agent = SuperAgent()
langgraph_agent = GraphSuperAgent()

@router.post("/langchain/chat")
async def chat_langchain(message: str):
    try:
        response = await langchain_agent.run(message)
        return {"response": response, "type": "langchain"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langgraph/chat")
async def chat_langgraph(message: str):
    try:
        response = await langgraph_agent.run(message)
        return {"response": response, "type": "langgraph"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare/chat")
async def compare_agents(message: str):
    try:
        langchain_response = await langchain_agent.run(message)
        langgraph_response = await langgraph_agent.run(message)
        
        return {
            "langchain": langchain_response,
            "langgraph": langgraph_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 