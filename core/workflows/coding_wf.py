from .abstract_wf import BaseWorkflow
from langgraph.graph import StateGraph, END
from agents.coding_agent import CodingAgent
from core.state import WorkflowStep, CodeWorkflowState


class CodeWorkflow(BaseWorkflow):
    def __init__(self, agent: CodingAgent):
        super().__init__(agent)

    def build_graph(self) -> StateGraph:
        builder = StateGraph(CodeWorkflowState)

        builder.add_node("analyze", self._analyze_task)
        builder.add_node("generate_code", self._generate_code)
        builder.add_node("review_code", self._review_code)
        builder.add_node("reflect", self._reflect)
        builder.add_node("finalize", self._finalize)
        
        
        builder.set_entry_point("analyze")

        builder.add_edge("analyze", "generate_code")
        builder.add_edge("generate_code", "review_code")
        builder.add_conditional_edges(
            "review_code",
            self._should_reflect,
            {
                "reflect": "reflect",
                "finalize": "finalize"
            }
        )
        builder.add_edge("reflect", "generate_code")
        builder.add_edge("finalize", END)
        
        return builder

    async def _analyze_task(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.ANALYSIS
            state.analysis = await self.agent.analyze(state.user_input)
        except Exception as e:
            raise e
        return state

    async def _generate_code(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.GENERATION
            if state.analysis:
                feedback = state.metadata.get("feedback", [])
                state.generated_code = await self.agent.generate(
                    state.analysis, feedback
                )
            state.error = None
        except Exception as e:
            raise e
        return state

    async def _review_code(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.REVIEW
            if state.analysis and state.generated_code:
                state.review = await self.agent.review(
                    state.generated_code, state.analysis
                )
        except Exception as e:
            raise e
        return state

    def _reflect(self, state: CodeWorkflowState) -> CodeWorkflowState:
        state.current_step = WorkflowStep.REFLECTION
        state.retry_count += 1
        
        if state.review and not state.review.approved:
            state.metadata["feedback"] = state.review.suggestions
            state.needs_retry = True
        return state
    
    def _should_reflect(self, state: CodeWorkflowState) -> str:
        if state.review is None:
            return "finalize"
        
        if not state.review.approved and state.retry_count < state.max_retries:
            return "reflect"
        return "finalize"    

    def _finalize(self, state: CodeWorkflowState) -> CodeWorkflowState:
        state.current_step = WorkflowStep.FINAL
        if state.analysis and state.generated_code and state.review:
            state.final_result = {
                "analysis": state.analysis,
                "code": state.generated_code,
                "review": state.review
            }
        return state

