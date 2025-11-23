from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .abstract_wf import BaseWorkflow


class ResearchWorkflowState(BaseModel):
    research_topic: str = ""
    research_depth: str = "deep"  # deep | quick
    findings: List[Dict] = []
    analysis: Optional[str] = None


class ResearchWorkflow(BaseWorkflow):
    def __init__(self, agent: Any):
        super().__init__(agent)
        # В будущем здесь может быть вызов CodingWorkflow!

    def build_graph(self):
        # Заглушка - будет реализована позже
        from langgraph.graph import StateGraph

        builder = StateGraph(ResearchWorkflowState)
        # ... здесь будет граф research workflow
        return builder

    async def run(self, user_input: str):
        return f"Research workflow выполнил исследование по теме: {user_input}. [Это заглушка]"
