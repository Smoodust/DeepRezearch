import os
from abc import ABC, abstractmethod
from typing import TypedDict

from jinja2 import (Environment, FileSystemLoader, Template, TemplateNotFound,
                    select_autoescape)
from langgraph.graph import StateGraph
from loguru import logger


class BaseAgentState(TypedDict):
    workflow_input: str


class BaseAgentOutput(TypedDict):
    output: str


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
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def purpose(self) -> str:
        raise NotImplementedError()

    @property
    def additional_input_prompt(self) -> str:
        raise NotImplementedError()

    @property
    def examples_input_prompt(self) -> str:
        raise NotImplementedError()

    @property
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    def _load_template(self, name: str, default: str = "default") -> Template:
        if not self.jinja_env:
            raise RuntimeError("Jinja environment not initialized")

        try:
            return self.jinja_env.get_template(name)
        except TemplateNotFound:
            logger.error(f"Template NOT FOUND: {name}")
            raise
        except Exception:
            logger.exception(f"Template FOUND but FAILED to load: {name}")
            raise

    async def run(self, state: BaseAgentState) -> BaseAgentOutput:
        return await self.compiled_graph.ainvoke(state)  # type: ignore
