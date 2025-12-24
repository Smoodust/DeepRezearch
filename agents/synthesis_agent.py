from typing import Any, Dict, List, Literal, Optional, Type, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import BaseMessage
from loguru import logger
from pydantic import BaseModel, Field

from core.template_manager import TemplateManager

from .base_agent import BaseAgent, BaseAgentOutput, BaseAgentState


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


class SynthesisAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
        self.model_final_answer = self.model.with_structured_output(
            SynthesisStructuredOutput
        )

        self._system_prompt = None
        self._synth_input_tpl = None

    @property
    def name(self) -> str:
        return "RESPONSE_SYNTHESIZER"

    @property
    def purpose(self) -> str:
        purpose = """Synthesizes multi-source information into coherent, well-structured final responses
Capabilities:
- Integrates information from multiple agent outputs
- Formats responses for different audience types
- Structures complex information logically
- Maintains conversational tone and clarity
- Adheres to specified format requirements
 Use when:
- Preparing final responses from multi-agent workflows
- Combining research findings with computational result
- Handling conversational greetings and informal queries"""
        return purpose

    @property
    def additional_input_prompt(self) -> str:
        return """- workflow_input: Specify synthesis requirements: "Synthesize the [information] from provided [context] into [response_format] addressing [user's_original_query]. Focus on [key_points] and structure for [audience_type]."
- context: Include messages containing:
  - User's original question/request (usually first message)
  - Agent outputs with results/data (identified by their IDs)
  - Any formatting or tone requirements
  - NEVER include intermediate thinking or routing decisions"""

    @property
    def get_input_model(self) -> Type[BaseModel]:
        return SynthesisAnalysis

    async def synthesis(self, state: SynthesisAgentState) -> BaseAgentOutput:
        if self._system_prompt is None:
            self._system_prompt = self._load_template(
                "synthesis_agent/SYNTHESIS_SYSTEM_PROMPT.jinja"
            ).render()

        if self._synth_input_tpl is None:
            self._synth_input_tpl = self._load_template()

        messages = state["messages"]
        messages.append(
            SystemMessage(
                TemplateManager().render_template(
                    "synthesis_agent/SYNTHESIS_SYSTEM_PROMPT.jinja"
                )
            )
        )
        messages.append(
            HumanMessage(
                TemplateManager().render_template(
                    "synthesis_agent/SYNTHESIS_INPUT.jinja",
                    workflow_input=state["workflow_input"],
                )
            )
        )
        response: SynthesisStructuredOutput = await self.model_final_answer.ainvoke(
            messages
        )
        logger.info(f"[synthesis]: {response.final_answer}")

        return {"output": response}  # type: ignore

    def build_graph(self) -> StateGraph:
        try:
            builder = StateGraph(
                OverallSynthesisState,
                input_schema=SynthesisAgentState,
                output_schema=BaseAgentOutput,
            )

            builder.add_node("synthesis", self.synthesis)

            builder.set_entry_point("synthesis")
            builder.add_edge("synthesis", END)

            logger.success(f"[{self.name}] ✅ Workflow граф успешно построен")
            return builder

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка при построении графа: {e}")
            raise

    async def run(self, state: SynthesisAgentState) -> BaseAgentOutput:
        return await self.compiled_graph.ainvoke(state)  # type: ignore
