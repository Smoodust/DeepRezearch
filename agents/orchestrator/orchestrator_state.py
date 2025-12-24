from typing import TypedDict

from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgentStrcturedInput


class OrchestratorTypeDecision(BaseModel):
    """Model for orchestrator routing decision"""

    thinking: str = Field(description="Reasoning behind the decision")
    workflow_type: str = Field(description="Type of workflow")
    confidence: float = Field(description="Confidence score from 0 to 1")


class OrchestratorInputDecision(BaseModel):
    """Model for orchestrator routing decision"""

    thinking: str = Field(description="Reasoning behind the decision")
    workflow_input: str = Field(
        description="Command or request that this workflow should do"
    )
    context: list[int] = Field(
        description="List of ids messages that would be put into context"
    )


class OrchestratorState(TypedDict):
    user_input: str

    last_judged_workflow_type: str
    last_judged_workflow_input: BaseAgentStrcturedInput
    last_judged_workflow_context: list[int]
    messages: list[BaseMessage]
