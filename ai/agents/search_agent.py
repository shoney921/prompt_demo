from langchain_community.tools import Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from .base_agent import BaseSubAgent

class SearchAgent(BaseSubAgent):
    def setup_tools(self):
        self.tools = [
            Tool(
                name="웹검색",
                func=SerpAPIWrapper().run,
                description="웹에서 정보를 검색하는 도구"
            ),
            Tool(
                name="위키피디아",
                func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
                description="위키피디아에서 정보를 검색해야 할 때 사용하는 도구"
            )
        ] 