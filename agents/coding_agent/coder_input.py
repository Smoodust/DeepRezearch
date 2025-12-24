from enum import Enum
from typing import List, Literal

from pydantic import Field

from ..base_agent import BaseAgentStrcturedInput


class IntentEnum(str, Enum):
    """Possible intents of a user request."""

    CODE_GENERATION = "code_generation"
    СALCULATION = "make_calculations"
    OTHER = "other"


class ComplexityEnum(str, Enum):
    """Estimated complexity of the task."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CodingUserInput(BaseAgentStrcturedInput):
    # The fields are defined in the order the LLM should reason through them.
    workflow_type: Literal["PYTHON_EXECUTOR"]

    input: str = Field(description="Full task")
    step2_intent: IntentEnum = Field(description="The primary intent of the request.")

    step6_test_required: bool = Field(
        default=False, description="Whether the task likely requires writing tests."
    )

    selected_context_ids: List[int] = Field(
        default_factory=list,
        description="List of ids from context that should be given to agent",
    )

    def to_string(self) -> str:
        return f"""
        #TASK
        {self.input}

        # Testing Required
        {"Yes" if self.step6_test_required else "No"}
        """
