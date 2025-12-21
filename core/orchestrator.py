from typing import Annotated, Dict, TypedDict, cast
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import BaseMessage, add_messages
from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from agents.synthesis_agent import SynthesisAgent, SynthesisAgentState


class OrchestratorDecision(BaseModel):
    """Model for orchestrator routing decision"""

    thinking: str = Field(description="Reasoning behind the decision")
    workflow_type: str = Field(description="Type of workflow")
    workflow_input: str = Field(
        description="Command or request that this workflow should do"
    )
    confidence: float = Field(description="Confidence score from 0 to 1")


class OrchestratorState(TypedDict):
    user_input: str

    last_judged_workflow_type: str
    last_judged_workflow_input: str
    messages: Annotated[list[BaseMessage], add_messages]


class WorkflowOrchestrator:
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        templates_dir: str | None = "prompts/orchestrator",
    ):
        self.base_model = init_chat_model(model_name, model_provider="ollama")
        self.model = self.base_model.with_structured_output(OrchestratorDecision)
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
            self.jinja_env = None

        self._system_template = None
        self._analysis_template = None
        self._decision_schema = None

    def register_workflow(self, workflow: BaseAgent):
        """Register a workflow with the orchestrator"""
        self.workflows[workflow.name] = workflow

    def _load_template(self, name: str, default: str) -> Template:
        if self.jinja_env:
            try:
                return self.jinja_env.get_template(name)
            except Exception:
                logger.warning(
                    f"Template {name} not found in {self.templates_dir}, using default."
                )
        return Template(default)

    @property
    def system_prompt(self) -> str:
        workflows_list = []
        for index, x in enumerate(self.workflows.values()):
            workflows_list.append(f"{index+1}. {x.name} - {x.purpose}")
        workflows_list = "\n".join(workflows_list)

        workflow_variants = "|".join([x for x in self.workflows.keys()])

        if self._system_template is None:
            default_system = """<DEFAULT SYSTEM PROMPT FALLBACK - you can replace with file prompts/orchestrator/system_prompt.j2>"""
            self._system_template = self._load_template(
                "SYSTEM_PROMPT.jinja", default_system
            )

        if self._decision_schema is None:
            default_schema = '{"thinking": "{{ thinking_placeholder }}","workflow_type": "{{ workflow_variants }}","workflow_input": "{{ workflow_input_placeholder }}","confidence": {{ confidence_placeholder }}}'
            self._decision_schema = self._load_template(
                "DECISION_SCHEMA.jinja", default_schema
            )

        decision_schema_rendered = self._decision_schema.render(
            thinking_placeholder="{{thinking}}",
            workflow_variants=workflow_variants,
            workflow_input_placeholder="{{workflow_input}}",
            confidence_placeholder="{{confidence}}",
        )

        rendered = self._system_template.render(
            workflows_list=workflows_list,
            workflow_variants=workflow_variants,
            decision_schema=decision_schema_rendered,
        )
        return rendered

    def render_analysis_prompt(
        self, user_input: str, history_messages: list[BaseMessage]
    ):
        agent_outputs = [
            m.content for m in history_messages if isinstance(m, AIMessage)
        ]
        history_summary = ""
        if agent_outputs:
            formatted = []
            for out in agent_outputs:
                if len(out) > 500:
                    formatted.append(f"• {out[:500]}...")
                else:
                    formatted.append(f"• {out}")
            history_summary = "\n".join(formatted)

        if self._analysis_template is None:
            default_analysis = "USER ORIGINAL REQUEST: {{ user_input }}\n\n{% if history_summary %}PREVIOUS AGENT OUTPUTS:\n{{ history_summary }}\n{% endif %}\nBased on the user request and any previous agent outputs, decide what should happen next.\n\nThink step by step and return ONLY JSON:\n"
            self._analysis_template = self._load_template(
                "ANALYSIS_PROMPT.jinja", default_analysis
            )

        return self._analysis_template.render(
            user_input=user_input, history_summary=history_summary
        )

    async def analyze_request(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze the request and make routing decision"""
        analysis_prompt = self.render_analysis_prompt(state["user_input"], state["messages"])

        try:
            # Формируем запрос к модели
            request_messages = [
                SystemMessage(content=self.system_prompt),
                # Добавляем историю сообщений
                *state["messages"],
                # Добавляем новый промпт для анализа
                SystemMessage(content=analysis_prompt),
            ]

            decision_data = await self.model.ainvoke(request_messages)

            logger.success(
                f"[orchestrator] made decision: {decision_data.workflow_type} with confidence {decision_data.confidence}"
            )
            logger.info(f"Decision thinking: {decision_data.thinking}")

        except Exception as e:
            logger.error(f"Analysis error: {repr(e)}")
            decision_data = OrchestratorDecision(
                thinking=f"Analysis error: {str(e)}",
                workflow_type="synthesis",
                workflow_input="Synthesize the information gathered so far",
                confidence=0.5,
            )

        # Сохраняем решение в истории сообщений
        decision_message = AIMessage(decision_data.model_dump_json(indent=4))

        return {
            "user_input": state["user_input"],
            "messages": state["messages"] + [decision_message],
            "last_judged_workflow_type": decision_data.workflow_type,
            "last_judged_workflow_input": decision_data.workflow_input,
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
            "messages": [],
        }

        max_iterations = 5  # Защита от бесконечного цикла
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[orchestrator] Iteration {iteration}")

            state = await self.analyze_request(state)
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
                workflow_output = await self.workflows[current_workflow_type].run(
                    {"workflow_input": state["last_judged_workflow_input"]}
                )

                # Сохраняем вывод агента в историю
                agent_output = workflow_output.get("output", "No output from agent")
                state["messages"].append(
                    AIMessage(f"Agent {current_workflow_type} output:\n{agent_output}")
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
        synth_state: SynthesisAgentState = {
            "workflow_input": f"Synthesize a response to: {user_input}",
            "messages": state["messages"],
        }

        final_result = await synth_agent.run(synth_state)
        return final_result["output"]
