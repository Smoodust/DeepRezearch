from typing import Annotated, Dict, TypedDict, cast
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage
from agents.base_agent import BaseAgent, BaseAgentState
from agents.synthesis_agent import SynthesisAgent, SynthesisAgentState
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.types import Overwrite
from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent

from ..agents.prompts import workflow_system_prompt


class OrchestratorDecision(BaseModel):
    """Model for orchestrator routing decision"""

    workflow_type: str = Field(description="Type of workflow")
    reasoning: str = Field(description="Reasoning behind the decision")
    confidence: float = Field(description="Confidence score from 0 to 1")
    workflow_input: str = Field(
        description="Command or request that this workflow should do"
    )


class OrchestratorState(TypedDict):
    user_input: str

    last_judged_workflow_type: str
    last_judged_workflow_input: str
    messages: Annotated[list[BaseMessage], add_messages]


class WorkflowOrchestrator:
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = init_chat_model(model, model_provider="ollama")
        self.workflows: Dict[str, BaseAgent] = {}

    def register_workflow(self, workflow: BaseAgent):
        """Register a workflow with the orchestrator"""
        self.workflows[workflow.name] = workflow

    @property
    def workflows_list(self) -> str:
        results = []
        for index, x in enumerate(self.workflows.values()):
            results.append(f"{index+1}. {x.name} - {x.purpose}")
        return "\n".join(results)

    @property
    def workflow_variants(self) -> str:
        return "|".join([x for x in self.workflows.keys()])

    @property
    def system_prompt(self) -> str:
        return workflow_system_prompt.format(
            workflows_list=self.workflows_list, workflow_variants=self.workflow_variants
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
            logger.error(f"{e}")
            decision_data = OrchestratorDecision(
                workflow_type="synthesis",
                workflow_input="Make summary of previous text into markdown",
                reasoning=f"Analysis error: {str(e)}",
                confidence=0.5,
            )
            # Fallback decision on error с ВСЕМИ обязательными полями
        return {
            "messages": [decision_data.model_dump_json(indent=4)],
            "last_judged_workflow_type": Overwrite(value=decision_data.workflow_type),
            "last_judged_workflow_input": Overwrite(value=decision_data.workflow_input)
        } # type: ignore

    async def process_request(
        self, user_input: str
    ) -> str:
        """Main method for processing requests"""

        if "synthesis" in self.workflows:
            raise Exception("There should be agent for final answer.")
        
        state: OrchestratorState = {"user_input": user_input, "last_judged_workflow_type": "null", "last_judged_workflow_input": "", "messages": []}

        while True:
            state = await self.analyze_request(state)
            workflow_state: BaseAgentState = 
            if state["last_judged_workflow_type"] == "synthesis":
                break
            workflow_output = await self.workflows[state["last_judged_workflow_type"]].run({"workflow_input": state["last_judged_workflow_input"]})
            state["messages"].append(AIMessage(workflow_output["output"]))
        
        synth_agent: SynthesisAgent = cast(SynthesisAgent, self.workflows["synthesis"])
        synth_state: SynthesisAgentState = {"workflow_input": state["last_judged_workflow_input"], "messages": state["messages"]}
        return (await synth_agent.run(synth_state))["output"]
