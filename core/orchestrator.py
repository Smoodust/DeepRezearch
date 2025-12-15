from typing import Annotated, Dict, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import BaseMessage, add_messages
from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from agents.synthesis_agent import SynthesisAgent, SynthesisAgentState

workflow_system_prompt = """
You are the Master Control Unit (MCU) for a specialized AI team. Your role is SOLELY to analyze incoming requests and route them to the appropriate specialist agent.

CORE SPECIALIST AGENTS:
- You are a DISPATCHER, not a problem solver
- You NEVER execute tasks yourself
- You NEVER write code, perform calculations, or conduct research
- Your ONLY output is a routing decision in JSON format

AVAILABLE WORKFLOWS:
{workflows_list}

DECISION CRITERIA:
- Technical/calculation tasks → PYTHON_EXECUTOR
- Information/research tasks → WEB_RESEARCHER
- You received enough information to respond to the user → RESPONSE_SYNTHESIZER
- Mixed tasks → Choose primary need, one agent at a time
- Unclear requests → Ask for clarification via workflow_input

CRITICAL CONSTRAINTS:
- NEVER attempt to answer questions yourself
- NEVER combine agent capabilities in one decision
- ALWAYS output valid JSON with NO additional text
- If uncertain, assign with lower confidence and clear workflow_input

CRITICAL: You MUST respond with ONLY a JSON object in this exact format:
{{
    "thinking": "Thoughts and brief explanation of classification",
    "workflow_type": "{workflow_variants}",
    "workflow_input": "command or request that this workflow should do",
    "confidence": 0.0-1.0
}}


DO NOT add any other text, explanations, or answers before or after the JSON.
DO NOT use markdown formatting.
DO NOT include code examples.
"""


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
    def __init__(self, model_name: str = "qwen3:0.6b"):
        self.base_model = init_chat_model(model_name, model_provider="ollama")
        self.model = self.base_model.with_structured_output(OrchestratorDecision)
        self.workflows: Dict[str, BaseAgent] = {}

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

        return workflow_system_prompt.format(
            workflows_list=workflows_list, workflow_variants=workflow_variants
        )

    async def analyze_request(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze the request and make routing decision"""

        # Создаем сводку о том, что уже было сделано
        history_summary = ""
        if state["messages"]:
            # Извлекаем только выводы агентов (AIMessage)
            agent_outputs = [
                msg.content for msg in state["messages"] if isinstance(msg, AIMessage)
            ]
            if agent_outputs:
                history_summary = f"""
    PREVIOUS AGENT OUTPUTS:
    {chr(10).join([f"• {output[:500]}..." if len(output) > 500 else f"• {output}" for output in agent_outputs])}
    """

        analysis_prompt = f"""
    USER ORIGINAL REQUEST: {state["user_input"]}

    {history_summary}

    Based on the user request and any previous agent outputs, decide what should happen next.

    Options:
    1. If enough information has been gathered to answer the user → Choose SYNTHESIS agent
    2. If more information is needed → Choose appropriate agent (WEB_RESEARCHER for research, PYTHON_EXECUTOR for calculations)
    3. If the current agent output needs further processing → Choose appropriate follow-up agent

    Think step by step and return ONLY JSON:
    """

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
