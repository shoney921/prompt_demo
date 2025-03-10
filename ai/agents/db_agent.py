from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import ReadFileTool, WriteFileTool
from .base_agent import BaseSubAgent

class DBAgent(BaseSubAgent):
    def setup_tools(self):
        self.tools = [
            PythonREPLTool(
                name="파이썬_DB실행기",
                description="데이터베이스 관련 파이썬 코드를 실행하는 도구"
            ),
            ReadFileTool(
                name="DB파일_읽기",
                description="데이터베이스 파일을 읽는 도구"
            ),
            WriteFileTool(
                name="DB파일_쓰기",
                description="데이터베이스 파일에 쓰는 도구"
            )
        ] 