from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from agents.coding_agent.coding_agent import CodingAgent

from ..base_agent import BaseAgent, BaseAgentBuilder
from .code_interfaces import *


@dataclass
class CodingAgentBuilder(BaseAgentBuilder):
    """Configuration for the coding agent."""

    model_name: str
    analyzer: Optional[ICodeAnalyzer] = None
    generator: Optional[ICodeGenerator] = None
    reviewer: Optional[ICodeReviewer] = None

    chat: type[BaseChatModel] = ChatOllama
    max_retries: int = 3
    approval_threshold: int = 6
    max_feedback_items: int = 3
    max_stored_feedback: int = 10
    temperature: float = 0.1
    num_predict: int = 2048

    name: str = "PYTHON_EXECUTOR"
    purpose: str = """
    Write and Execute Python code for calculations, data processing, and algorithmic tasks
    Capabilities:
    - Python REPL environment with standard libraries only
    - Mathematical computations and data transformations
    - Script execution for well-defined programming tasks
    - No internet access or external libraries
    Use when: Task requires computation, data manipulation, or algorithmic processing
    """

    def validate(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if not 0 <= self.approval_threshold <= 10:
            raise ValueError("approval_threshold must be between 0 and 10")

    def build(self) -> BaseAgent:
        return CodingAgent(
            self.name,
            self.purpose,
            self.model_name,
            self.analyzer,
            self.generator,
            self.reviewer,
            self.chat,
            self.max_retries,
            self.approval_threshold,
            self.max_feedback_items,
            self.max_stored_feedback,
            self.temperature,
            self.num_predict,
        )
