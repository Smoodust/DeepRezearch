from abc import ABC, abstractmethod

from langgraph.graph import StateGraph

from agents.base_agent import BaseAgent


class BaseWorkflow(ABC):
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self._compiled_graph = None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass

    @property
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    async def run(self, state):
        return await self.compiled_graph.ainvoke(state)
