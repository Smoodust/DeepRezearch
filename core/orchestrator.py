import json
import os
from typing import Dict, TypedDict, cast

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import BaseMessage
from loguru import logger
from pydantic import BaseModel, Field
from template_manager import TemplateManager

from agents.base_agent import BaseAgent, BaseAgentStrcturedInput
from agents.synthesis_agent import SynthesisAgent, SynthesisAgentState


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


class WorkflowOrchestrator:
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        templates_dir: str | None = "prompts/orchestrator",
    ):
        self.base_model = init_chat_model(model_name, model_provider="ollama")
        self.model_workflow_type = self.base_model.with_structured_output(
            OrchestratorTypeDecision
        )
        self.workflows: Dict[str, BaseAgent] = {}

        self.template_dir = templates_dir
        if templates_dir and os.path.isdir(templates_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                autoescape=select_autoescape(enabled_extensions=()),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            raise Exception("There is no prompts folder")

    def register_workflow(self, workflow: BaseAgent):
        """Register a workflow with the orchestrator"""
        self.workflows[workflow.name] = workflow

    @property
    def system_prompt(self) -> str:
        workflows_list = []
        for index, x in enumerate(self.workflows.values()):
            workflows_list.append(f"{index+1}. {x.name} - {x.purpose}")
        workflows_list = "\n".join(workflows_list)

        workflow_variants = "|".join([x for x in self.workflows.keys()])

        rendered = TemplateManager().render_template(
            "WORKFLOW_TYPE_PROMPT.jinja",
            workflows_list=workflows_list,
            workflow_variants=workflow_variants,
        )
        return rendered

    async def decide_workflow_type(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze the request and make routing decision"""

        analysis_prompt = TemplateManager().render_template(
            "ANALYSIS_PROMPT.jinja", user_input=state["user_input"]
        )

        try:
            # Формируем запрос к модели
            request_messages = [
                SystemMessage(content=self.system_prompt),
                # Добавляем историю сообщений
                *state["messages"],
                # Добавляем новый промпт для анализа
                SystemMessage(content=analysis_prompt),
            ]

            decision_data: OrchestratorTypeDecision = await self.model_workflow_type.ainvoke(request_messages)  # type: ignore

            logger.success(
                f"[orchestrator] made decision: {decision_data.workflow_type} with confidence {decision_data.confidence}"
            )
            logger.info(f"Decision thinking: {decision_data.thinking}")

        except Exception as e:
            logger.error(f"Analysis error: {repr(e)}")
            decision_data = OrchestratorTypeDecision(
                thinking=f"Analysis error: {str(e)}",
                workflow_type="synthesis",
                confidence=0.5,
            )

        # Сохраняем решение в истории сообщений
        decision_message = AIMessage(decision_data.model_dump_json(indent=4))

        return {
            "user_input": state["user_input"],
            "messages": state["messages"] + [decision_message],
            "last_judged_workflow_type": decision_data.workflow_type,
            "last_judged_workflow_input": "",
            "last_judged_workflow_context": [],
        }

    async def decide_workflow_input(
        self, state: OrchestratorState
    ) -> OrchestratorState:
        """Analyze the request and make routing decision"""
        current_agent: BaseAgent = self.workflows[state["last_judged_workflow_type"]]

        analysis_prompt = TemplateManager().render_template(
            "WORKFLOW_INPUT_PROMPT.jinja",
            messages=state["messages"],
            user_input=state["user_input"],
            chosen_workflow=state["last_judged_workflow_type"],
        )

        try:
            # Формируем запрос к модели
            request_messages = [
                # Добавляем историю сообщений
                *state["messages"],
                # Добавляем новый промпт для анализа
                SystemMessage(content=analysis_prompt),
            ]

            model_workflow_input = self.base_model.with_structured_output(
                current_agent.get_input_model
            )
            decision_data: BaseAgentStrcturedOutput = await model_workflow_input.ainvoke(request_messages)  # type: ignore
            logger.info(f"Decision result: {decision_data.model_dump_json()}")

            result_ids = []
            for ids in decision_data.selected_context_ids:  # type: ignore
                ids = cast(int, ids)
                if ids >= 0 and ids < len(state["messages"]):
                    result_ids.append(ids)
            decision_data.selected_context_ids = result_ids
            logger.info(f"Cleaned context: {decision_data.selected_context_ids}")

        except Exception as e:
            logger.error(f"Analysis error: {repr(e)}")
            decision_data = BaseAgentStrcturedInput(selected_context_ids=[])

        # Сохраняем решение в истории сообщений
        decision_message = AIMessage(decision_data.model_dump_json(indent=4))

        return {
            "user_input": state["user_input"],
            "messages": state["messages"] + [decision_message],
            "last_judged_workflow_type": state["last_judged_workflow_type"],
            "last_judged_workflow_input": decision_data,
            "last_judged_workflow_context": decision_data.selected_context_ids,
        }

    @logger.catch
    async def process_request(self, user_input: str) -> str:
        """Main method for processing requests"""

        # Находим ключ для synthesis агента
        synthesis_key = None
        for name, agent in self.workflows.items():
            if isinstance(agent, SynthesisAgent):
                synthesis_key = name
                break

        if not synthesis_key:
            raise Exception("There should be agent for final answer.")

        state: OrchestratorState = {
            "user_input": user_input,
            "last_judged_workflow_type": "",
            "last_judged_workflow_input": "",
            "last_judged_workflow_context": [],
            "messages": [],
        }

        max_iterations = 5  # Защита от бесконечного цикла
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[orchestrator] Iteration {iteration}")

            state = await self.decide_workflow_type(state)
            state = await self.decide_workflow_input(state)
            current_workflow_type = state["last_judged_workflow_type"]

            # Проверяем, не запрашивает ли orchestrator synthesis
            if current_workflow_type == synthesis_key:
                logger.info(f"[orchestrator] Decision to use synthesis agent")
                break

            # Проверяем, существует ли запрошенный workflow
            if current_workflow_type not in self.workflows:
                logger.error(
                    f"[orchestrator] Unknown workflow type: {current_workflow_type}"
                )
                # По умолчанию переходим к synthesis
                state["last_judged_workflow_type"] = synthesis_key
                state["last_judged_workflow_input"] = (
                    "Process the available information due to unknown workflow request"
                )
                break

            # Запускаем выбранный агент
            logger.info(
                f"[orchestrator] Executing {current_workflow_type} with input: {state['last_judged_workflow_input']}"
            )

            try:
                workflow_input = "Context:\n" + "\n".join([state["messages"][ids].content for ids in state["last_judged_workflow_context"]])  # type: ignore

                workflow_input += (
                    f"TASK:\n{state["last_judged_workflow_input"].to_string()}"
                )

                logger.info(f"PROMPT: {workflow_input}")

                workflow_output = await self.workflows[current_workflow_type].run(
                    {"workflow_input": workflow_input}
                )

                # Сохраняем вывод агента в историю
                agent_output = workflow_output.get("output", "No output from agent")
                agent_message_id = len(state["messages"])
                state["messages"].append(
                    AIMessage(
                        json.dumps(
                            {
                                "id": agent_message_id,
                                "from": current_workflow_type,
                                "output": agent_output,
                            },
                            indent=4,
                        )
                    )
                )
                logger.info(
                    f"[orchestrator] Agent {current_workflow_type} completed successfully"
                )

            except Exception as e:
                logger.error(
                    f"[orchestrator] Error executing {current_workflow_type}: {e}"
                )
                state["messages"].append(
                    AIMessage(f"Error executing {current_workflow_type}: {str(e)}")
                )

        # В конце всегда используем synthesis агента для формирования ответа
        synth_agent: SynthesisAgent = cast(
            SynthesisAgent, self.workflows[synthesis_key]
        )

        synthesis_context = "\n".join(
            [
                state["messages"][ids].content
                for ids in state["last_judged_workflow_context"]
            ]
        )

        synth_state: SynthesisAgentState = {
            "workflow_input": f"Synthesize a response to: {user_input}. With context: {synthesis_context}",
            "messages": state["messages"],
        }

        final_result = await synth_agent.run(synth_state)
        return final_result["output"]
