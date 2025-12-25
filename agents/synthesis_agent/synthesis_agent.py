from typing import Type

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel

from core.template_manager import TemplateManager

from ..base_agent import BaseAgent, BaseAgentOutput
from .synthesis_state import (OverallSynthesisState, SynthesisAgentState,
                              SynthesisAnalysis, SynthesisStructuredOutput)


class SynthesisAgent(BaseAgent):
    def __init__(
        self, name: str, purpose: str, additional_input_prompt: str, model_name: str
    ):
        super().__init__()

        self.model_name = model_name
        self.model = init_chat_model(self.model_name, model_provider="ollama")
        self.model_final_answer = self.model.with_structured_output(
            SynthesisStructuredOutput
        )

        self._name = name
        self._purpose = purpose
        self._additional_input_prompt = additional_input_prompt

    @property
    def name(self) -> str:
        return self._name

    @property
    def purpose(self) -> str:
        return self._purpose

    @property
    def additional_input_prompt(self) -> str:
        return self._additional_input_prompt

    @property
    def get_input_model(self) -> Type[BaseModel]:
        return SynthesisAnalysis

    async def synthesis(self, state: SynthesisAgentState) -> BaseAgentOutput:
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
        response: SynthesisStructuredOutput = await self.model_final_answer.ainvoke(messages)  # type: ignore
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
