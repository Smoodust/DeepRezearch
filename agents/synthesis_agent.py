from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import BaseMessage
from loguru import logger
from pydantic import BaseModel

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
        return """workflow_input: MUST be a synthesis instruction. Include:
- Desired output format: "Create a report...", "Write a summary..."
- Audience considerations: "Explain for beginners...", "Technical details..."
- Length/style preferences: "Concise...", "Detailed..."

context: MUST include ALL relevant information from previous agents:
- ALL relevant agent outputs (this is CRITICAL for synthesis)
- Original user request
- Any specific formatting requirements"""

    @property
    def examples_input_prompt(self) -> str:
        return """For RESPONSE_SYNTHESIZER (After previous agents have provided data):
```json
{
    "thinking": "RESPONSE_SYNTHESIZER needs all previous outputs plus the original request to create a cohesive answer",
    "workflow_input": "Synthesize the provided information into a clear, well-structured response for the user. Organize the information logically and maintain a professional tone.",
    "context": "Original user request: 'What's the latest SpaceX launch?' PYTHON_EXECUTOR output: 'N/A - Not used'. WEB_RESEARCHER output: 'Latest launch: Starlink Group 6-59 on December 15, 2024 from Cape Canaveral, carrying 23 Starlink satellites. Next launch: December 20, 2024 for Starlink Group 7-10.'"
}
```"""

    async def synthesis(self, state: SynthesisAgentState) -> BaseAgentOutput:
        if self._system_prompt is None:
            self._system_prompt = self._load_template(
                "synthesis_agent/SYNTHESIS_SYSTEM_PROMPT.jinja"
            ).render()

        if self._synth_input_tpl is None:
            self._synth_input_tpl = self._load_template(
                "synthesis_agent/SYNTHESIS_INPUT.jinja"
            )

        messages = state["messages"]
        messages.append(SystemMessage(self._system_prompt))
        messages.append(
            HumanMessage(
                self._synth_input_tpl.render(workflow_input=state["workflow_input"])
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
