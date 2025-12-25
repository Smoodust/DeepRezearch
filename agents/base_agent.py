from abc import ABC, abstractmethod
from typing import Protocol, Type, TypedDict

from langgraph.graph import StateGraph
from pydantic import BaseModel


class BaseAgentState(TypedDict):
    workflow_input: str


class BaseAgentOutput(TypedDict):
    output: str


class BaseAgentStrcturedInput(BaseModel):
    @abstractmethod
    def to_string(self) -> str: ...


class StringStructuredInput(BaseAgentStrcturedInput):
    output: str

    def to_string(self) -> str:
        return self.output


class BaseAgent(ABC):
    def __init__(
        self,
    ):
        self._compiled_graph = None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def purpose(self) -> str:
        pass

    @property
    @abstractmethod
    def get_input_model(self) -> Type[BaseModel]:
        pass

    @property
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    async def run(self, state: BaseAgentState) -> BaseAgentOutput:
        return await self.compiled_graph.ainvoke(state)  # type: ignore


class BaseAgentBuilder(Protocol):
    @abstractmethod
    def build(self) -> BaseAgent:
        pass
