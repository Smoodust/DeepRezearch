import os
from abc import ABC, abstractmethod
from typing import Type, TypedDict

from jinja2 import (Environment, FileSystemLoader, Template, TemplateNotFound,
                    select_autoescape)
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel


class BaseAgentState(TypedDict):
    workflow_input: str


class BaseAgentOutput(TypedDict):
    output: str


class BaseAgentStrcturedInput(BaseModel):
    def to_string(self) -> str:
        raise NotImplementedError()


class BaseAgent(ABC):
    def __init__(
        self,
        templates_dir: str | None = "prompts",
    ):
        self._compiled_graph = None

        self.template_dir = templates_dir
        if templates_dir and os.path.isdir(templates_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                autoescape=select_autoescape(enabled_extensions=()),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja_env = None

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
    @abstractmethod
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    async def run(self, state: BaseAgentState) -> BaseAgentOutput:
        return await self.compiled_graph.ainvoke(state)  # type: ignore
