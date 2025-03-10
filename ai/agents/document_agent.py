from langchain_community.tools import ReadFileTool, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from .base_agent import BaseSubAgent

class DocumentAnalysisAgent(BaseSubAgent):
    def setup_tools(self):
        self.tools = [
            ReadFileTool(
                name="문서_읽기",
                description="문서 파일을 읽는 도구"
            ),
            Tool(
                name="문서_분석",
                func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
                description="문서 내용을 분석하고 관련 정보를 검색하는 도구"
            )
        ] 