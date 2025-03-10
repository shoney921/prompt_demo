from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

from config import DEFAULT_MODEL
from agents.db_agent import DBAgent
from agents.document_agent import DocumentAnalysisAgent
from agents.search_agent import SearchAgent

class SuperAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL)
        
        # 서브 에이전트들 초기화
        self.db_agent = DBAgent(self.llm)
        self.doc_agent = DocumentAnalysisAgent(self.llm)
        self.search_agent = SearchAgent(self.llm)
        
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
            )
        ]
        
        # 슈퍼 에이전트 생성
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, input_text: str) -> str:
        return self.agent_executor.invoke({"input": input_text})["output"]

def test_super_agent():
    super_agent = SuperAgent()
    
    test_queries = [
        "prime_numbers.txt 파일에서 소수 목록을 읽어서 데이터베이스에 저장해주세요",
        "현재 디렉토리의 파일들을 분석해서 어떤 주제의 프로젝트인지 파악해주세요",
        "인공지능과 머신러닝의 차이점을 검색해서 알려주세요"
    ]
    
    for query in test_queries:
        try:
            print(f"\n질문: {query}")
            result = super_agent.run(query)
            print(f"응답: {result}")
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    test_super_agent()

