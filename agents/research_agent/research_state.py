from typing import List, Literal

from pydantic import Field

from ..base_agent import BaseAgentStrcturedInput


class WebResearchPlan(BaseAgentStrcturedInput):
    """Simplified schema for initial research planning (pre-web-search)."""

    workflow_type: Literal["WEB_RESEARCHER"]
    original_request: str
    interpreted_question: str
    research_angles: List[str]
    search_queries: List[str]
    selected_context_ids: List[int] = Field(
        description="List of context IDs that should be passed to synthesis agent"
    )

    def to_string(self) -> str:
        return f"""# Original request
{self.original_request}
# Interpreted question
{self.interpreted_question}
# Research angles
{"\n".join(self.research_angles)}
# Search queries
{"\n".join(self.search_queries)}"""
