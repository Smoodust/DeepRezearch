import re
from typing import Annotated, Dict, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import BaseMessage, add_messages
from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from agents.synthesis_agent import SynthesisAgent, SynthesisAgentState

workflow_system_prompt = """You are a routing assistant for a multi-agent system. Your ONLY job is to analyze user requests and classify them into the appropriate workflow category.

AVAILABLE WORKFLOWS:
{workflows_list}

IMPORTANT RULES:
- You are NOT a coding expert - DO NOT provide code solutions or technical implementations
- You are NOT a research assistant - DO NOT provide detailed analysis or research findings  
- You are ONLY a classifier - your response should ONLY contain the workflow decision and what this workflow should do
- NEVER write code, NEVER solve problems, NEVER provide detailed answers
- Your output MUST be valid JSON format ONLY - no additional text

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
        self.model = init_chat_model(model_name, model_provider="ollama")
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

    Classify this request and return ONLY JSON:
    """

        try:
            request = state["messages"]
            request += [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=analysis_prompt),
            ]

            response_chunks = []

            async for chunk in self.model.astream(request):
                if hasattr(chunk, "content"):
                    print(chunk.content, end="", flush=True)
                    response_chunks.append(chunk.content)

            response = "".join(response_chunks)

            # Очистка и парсинг ответа
            # content = response.content.strip()

            # Извлечение JSON из ответа

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match is None:
                raise Exception()

            decision_data = OrchestratorDecision.model_validate_json(json_match.group())

        except Exception as e:
            logger.error(repr(e))
            decision_data = OrchestratorDecision(
                workflow_type="synthesis",
                workflow_input="Make summary of previous text into markdown",
                reasoning=f"Analysis error: {str(e)}",
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

        if not "synthesis" in self.workflows:
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
