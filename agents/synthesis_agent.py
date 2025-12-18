from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import BaseMessage
from loguru import logger
from pydantic import BaseModel

from .base_agent import BaseAgent, BaseAgentOutput, BaseAgentState
from .prompts import SYNTHESIS_INPUT, SYNTHESIS_SYSTEM_PROMPT


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

    async def synthesis(self, state: SynthesisAgentState) -> BaseAgentOutput:
        messages = state["messages"]
        messages.append(SystemMessage(SYNTHESIS_SYSTEM_PROMPT))
        messages.append(
            HumanMessage(SYNTHESIS_INPUT.format(workflow_input=state["workflow_input"]))
        )
        response: SynthesisStructuredOutput = await self.model_final_answer.ainvoke(
            messages
        )
        logger.info(f"[synthesis]: {response}")

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
