from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import operator

from ai.config import DEFAULT_MODEL
from ai.agents.db_agent import DBAgent
from ai.agents.document_agent import DocumentAnalysisAgent
from ai.agents.search_agent import SearchAgent
from ai.agents.rag_agent import RAGTool

class GraphSuperAgent:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        
        # 서브 에이전트들 초기화
        self.db_agent = DBAgent(self.llm)
        self.doc_agent = DocumentAnalysisAgent(self.llm)
        self.search_agent = SearchAgent(self.llm)
        self.rag_tool = RAGTool()
        
        # 도구 설정
        self.tools = [
            Tool(
                name="DB_작업",
                func=self.db_agent.run,
                description="데이터베이스 관련 작업이 필요할 때 사용"
            ),
            Tool(
                name="문서_분석",
                func=self.doc_agent.run,
                description="문서를 읽고 분석해야 할 때 사용"
            ),
            Tool(
                name="정보_검색",
                func=self.search_agent.run,
                description="웹에서 정보를 검색해야 할 때 사용"
            ),
            Tool(
                name="문서_검색(RAG)",
                func=self.rag_tool._run,
                description="저장된 문서에서 관련 정보를 검색할 때 사용"
            )
        ]
        
        self.tool_executor = ToolExecutor(self.tools)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """
        LangGraph 워크플로우 생성
        
        # LangGraph vs LangChain 주요 차이점:
        # 1. StateGraph를 사용하여 명시적인 상태 관리와 전환을 정의
        #    - LangChain은 단순 체이닝 방식이지만, LangGraph는 상태 기반 워크플로우
        # 2. 비동기 실행 지원이 더 체계적
        # 3. 조건부 라우팅이 더 유연함 (operator.eq 등을 통한 엣지 조건 설정)
        # 4. 워크플로우의 시각화 및 디버깅이 용이
        """
        workflow = StateGraph(StateType=Dict)

        # route_message 함수는 상태 기반으로 다음 노드 결정
        # LangChain의 단순 체이닝과 달리, 상태에 따라 동적 라우팅 가능
        def route_message(state: Dict) -> str:
            message = state["message"]
            
            # RAG 키워드 확인
            rag_keywords = ["찾아줘", "검색해줘", "관련 정보", "문서에서", "문서 검색"]
            if any(keyword in message for keyword in rag_keywords):
                return "use_rag"
            
            # 다른 키워드 기반 라우팅
            if "데이터베이스" in message or "DB" in message:
                return "use_db"
            elif "문서" in message or "파일" in message:
                return "use_doc"
            elif "검색" in message:
                return "use_search"
            
            return "use_llm"

        # 각 노드는 상태를 입력받고 수정된 상태를 반환
        # LangGraph의 특징: 상태 객체를 통한 데이터 흐름 관리
        async def use_rag(state: Dict) -> Dict:
            context = await self.rag_tool._run(state["message"])
            response = await self.llm.ainvoke(
                [HumanMessage(content=f"다음 컨텍스트를 기반으로 답변해주세요:\n\n{context}\n\n질문: {state['message']}")]
            )
            state["response"] = response.content
            return state

        async def use_db(state: Dict) -> Dict:
            result = await self.db_agent.run(state["message"])
            state["response"] = result
            return state

        async def use_doc(state: Dict) -> Dict:
            result = await self.doc_agent.run(state["message"])
            state["response"] = result
            return state

        async def use_search(state: Dict) -> Dict:
            result = await self.search_agent.run(state["message"])
            state["response"] = result
            return state

        async def use_llm(state: Dict) -> Dict:
            response = await self.llm.ainvoke(
                [HumanMessage(content=state["message"])]
            )
            state["response"] = response.content
            return state

        # 워크플로우 구성
        # LangGraph의 특징: 명시적인 노드와 엣지 정의
        workflow.add_node("router", route_message)
        workflow.add_node("rag", use_rag)
        workflow.add_node("db", use_db)
        workflow.add_node("doc", use_doc)
        workflow.add_node("search", use_search)
        workflow.add_node("llm", use_llm)

        # 조건부 엣지 연결 - LangGraph의 강력한 기능
        workflow.add_edge("router", "rag", condition=operator.eq("use_rag"))
        workflow.add_edge("router", "db", condition=operator.eq("use_db"))
        workflow.add_edge("router", "doc", condition=operator.eq("use_doc"))
        workflow.add_edge("router", "search", condition=operator.eq("use_search"))
        workflow.add_edge("router", "llm", condition=operator.eq("use_llm"))

        # 종료 조건
        workflow.add_edge("rag", END)
        workflow.add_edge("db", END)
        workflow.add_edge("doc", END)
        workflow.add_edge("search", END)
        workflow.add_edge("llm", END)

        workflow.set_entry_point("router")
        
        return workflow.compile()

    async def run(self, input_text: str) -> str:
        """메시지 처리 및 응답 생성"""
        try:
            config = {"message": input_text}
            result = await self.workflow.ainvoke(config)
            return result["response"]
        except Exception as e:
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"

    def initialize_rag(self, documents: List[Document]):
        """RAG 도구 초기화"""
        self.rag_tool.initialize_vector_store(documents)

    async def load_all_documents(self):
        """모든 문서를 로드하고 RAG 시스템을 초기화합니다."""
        documents = self.rag_tool.load_documents()
        if documents:
            self.rag_tool.initialize_vector_store(documents)
            return f"{len(documents)}개의 문서가 로드되었습니다."
        return "로드할 문서가 없습니다."

    async def add_document(self, file_path: str):
        """새로운 문서를 추가합니다."""
        if self.rag_tool.add_document(file_path):
            return f"{os.path.basename(file_path)}가 성공적으로 추가되었습니다."
        return "문서 추가에 실패했습니다." 