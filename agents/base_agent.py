from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.tools import BaseTool


class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, context: Any) -> Any:
        pass