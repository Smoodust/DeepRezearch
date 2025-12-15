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

        analysis_prompt = f"""
    USER REQUEST: {state["user_input"]}
    HISTORY OF MESSAGES: {state["messages"]}

    Classify this request, messages and choose the next step. Return ONLY JSON:
    """

        try:
            request = state["messages"]
            request += [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=analysis_prompt),
            ]

            decision_data = await self.model.ainvoke(request)

            logger.success(
                f"[orchestrator] made decision: {decision_data.workflow_type} with confidence {decision_data.confidence}"
            )
            logger.info(decision_data)
            logger.debug(f"Decision details: {decision_data.thinking}")

        except Exception as e:
            logger.error(repr(e))
            decision_data = OrchestratorDecision(
                thinking=f"Analysis error: {str(e)}",
                workflow_type="synthesis",
                workflow_input="Make summary of previous text into markdown",
                confidence=0.5,
            )
        return {
            "user_input": state["user_input"],
            "messages": state["messages"]
            + [AIMessage(decision_data.model_dump_json(indent=4))],
            "last_judged_workflow_type": decision_data.workflow_type,
            "last_judged_workflow_input": decision_data.workflow_input,
        }  # type: ignore

    @logger.catch
    async def process_request(self, user_input: str) -> str:
        """Main method for processing requests"""

        synthesis_key = None
        for name, agent in self.workflows.items():
            if isinstance(agent, SynthesisAgent):
                synthesis_key = name
                break

        if not synthesis_key:
            raise Exception("There should be agent for final answer.")

        state: OrchestratorState = {
            "user_input": user_input,
            "last_judged_workflow_type": "null",
            "last_judged_workflow_input": "",
            "messages": [],
        }

        while True:
            state = await self.analyze_request(state)
            if state["last_judged_workflow_type"] == "synthesis":
                break
            workflow_output = await self.workflows[
                state["last_judged_workflow_type"]
            ].run({"workflow_input": state["last_judged_workflow_input"]})
            state["messages"].append(AIMessage(workflow_output["output"]))

        synth_agent: SynthesisAgent = cast(SynthesisAgent, self.workflows["synthesis"])
        synth_state: SynthesisAgentState = {
            "workflow_input": state["last_judged_workflow_input"],
            "messages": state["messages"],
        }
        return (await synth_agent.run(synth_state))["output"]
