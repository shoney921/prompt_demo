from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import BaseTool
from typing import List

class BaseSubAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = []
        self.agent = None
        self.agent_executor = None
        self.setup_tools()
        self.create_agent()

    def setup_tools(self):
        pass

    def create_agent(self):
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, input_text: str) -> str:
        return self.agent_executor.invoke({"input": input_text})["output"] 