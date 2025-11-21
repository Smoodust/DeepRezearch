from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.tools import BaseTool


class BaseAgent(ABC):
    def __init__(self, name: str, model: Any, tools: List[BaseTool]):
        self.name = name
        self.model = model
        self.tools = tools

        @abstractmethod
        def initialize():
            pass

        @abstractmethod
        async def process(self, state: GraphState):
            pass
