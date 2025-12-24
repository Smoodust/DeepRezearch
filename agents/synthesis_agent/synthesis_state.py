from langgraph.graph.message import BaseMessage
from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

from ..base_agent import  BaseAgentState


class SynthesisStructuredOutput(BaseModel):
    thinking: str
    final_answer: str


class SynthesisAgentState(BaseAgentState):
    workflow_input: str
    messages: list[BaseMessage]


class OverallSynthesisState(TypedDict):
    workflow_input: str
    messages: list[BaseMessage]

    output: str


class ContextRelevance(BaseModel):
    """Analysis of a specific context's relevance."""

    context_id: str = Field(description="ID of the context being analyzed")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Score from 0-1 indicating relevance to the query"
    )
    reason: str = Field(description="Reason for relevance score")
    context_type: Optional[str] = Field(
        default=None,
        description="Type of context (e.g., code_snippet, documentation, example, error_log)",
    )
    required: bool = Field(
        default=True, description="Whether this context is required for synthesis"
    )
    usage_purpose: List[
        Literal["evidence", "example", "reference", "counterpoint", "background"]
    ] = Field(
        default_factory=list, description="How this context should be used in synthesis"
    )


class SynthesisAnalysis(BaseModel):
    """Structured analysis of what context is needed for synthesis."""

    workflow_type: Literal["RESPONSE_SYNTHESIZER"]
    # Step 1: Understand the query
    step1_query_summary: str = Field(
        description="Concise summary of what the user is asking for"
    )

    # Step 2: Analyze each context's relevance
    step2_context_analysis: List[ContextRelevance] = Field(
        description="Analysis of each available context's relevance"
    )

    # Step 3: Select contexts for synthesis
    selected_context_ids: List[int] = Field(
        description="List of context IDs that should be passed to synthesis agent"
    )

    # Step 4: Output format guidance
    step4_output_format: Dict[str, Any] = Field(
        default_factory=lambda: {
            "structure": "logical",
            "include_headings": True,
            "code_blocks": True,
            "bullet_points": True,
        },
        description="Guidance for how to format the final output",
    )