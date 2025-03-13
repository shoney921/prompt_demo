from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from typing import List
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import os

from ai.config import DEFAULT_MODEL
from ai.agents.db_agent import DBAgent
from ai.agents.document_agent import DocumentAnalysisAgent
from ai.agents.search_agent import SearchAgent
from ai.agents.rag_agent import RAGTool

class SuperAgent:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        
        # 서브 에이전트들 초기화
        self.db_agent = DBAgent(self.llm)
        self.doc_agent = DocumentAnalysisAgent(self.llm)
        self.search_agent = SearchAgent(self.llm)
        self.rag_tool = RAGTool()
        
        # 슈퍼 에이전트용 도구 설정
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
        
        # 슈퍼 에이전트 생성
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def initialize_rag(self, documents: List[Document]):
        """RAG 도구 초기화"""
        self.rag_tool.initialize_vector_store(documents)
    
    async def _generate_response(self, prompt: str) -> str:
        """LLM을 사용하여 응답을 생성합니다."""
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트를 기반으로 정확하고 도움되는 답변을 제공합니다."),
            ("user", prompt)
        ])
        
        chain = chat_prompt | self.llm
        response = await chain.ainvoke({})
        return response.content
    
    async def process_message(self, message: str) -> str:
        """메시지 처리 및 응답 생성"""
        # RAG 관련 키워드 확인
        rag_keywords = ["찾아줘", "검색해줘", "관련 정보", "문서에서", "문서 검색"]
        
        try:
            if any(keyword in message for keyword in rag_keywords):
                # RAG 도구 사용
                context = self.rag_tool._run(message)
                prompt = f"""다음 컨텍스트를 기반으로 질문에 답변해주세요:
                
                컨텍스트:
                {context}
                
                질문: {message}
                """
                return await self._generate_response(prompt)
            
            # 일반적인 에이전트 실행
            result = await self.agent_executor.ainvoke({"input": message})
            return result["output"]
            
        except Exception as e:
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"
    
    async def run(self, input_text: str) -> str:
        """비동기 실행을 위한 메서드"""
        return await self.process_message(input_text)

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

# def test_super_agent():
#     super_agent = SuperAgent()
    
#     test_queries = [
#         "prime_numbers.txt 파일에서 소수 목록을 읽어서 데이터베이스에 저장해주세요",
#         "현재 디렉토리의 파일들을 분석해서 어떤 주제의 프로젝트인지 파악해주세요",
#         "인공지능과 머신러닝의 차이점을 검색해서 알려주세요"
#     ]
    
#     for query in test_queries:
#         try:
#             print(f"\n질문: {query}")
#             result = super_agent.run(query)
#             print(f"응답: {result}")
#         except Exception as e:
#             print(f"에러 발생: {str(e)}")

# if __name__ == "__main__":
#     test_super_agent()

