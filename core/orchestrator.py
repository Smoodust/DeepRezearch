import json
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class OrchestratorDecision(BaseModel):
    """Model for orchestrator routing decision"""

    workflow_type: str = Field(description="Type of workflow: 'coding', 'research'")
    reasoning: str = Field(description="Reasoning behind the decision")
    confidence: float = Field(description="Confidence score from 0 to 1")
    needs_additional_info: bool = Field(
        default=False, description="Whether additional information is needed"
    )


class WorkflowOrchestrator:
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = init_chat_model(model, model_provider="ollama")
        self.workflows: Dict[str, Any] = {}
        self._setup_prompts()

    def register_workflow(self, name: str, workflow: Any):
        """Register a workflow with the orchestrator"""
        self.workflows[name] = workflow

    def _setup_prompts(self):
        self.system_prompt = """You are a routing assistant for a multi-agent system. Your ONLY job is to analyze user requests and classify them into the appropriate workflow category.

AVAILABLE WORKFLOWS:
1. **coding** - For ACTUAL programming tasks: writing code, debugging, code review, implementation, optimization, leetcode tasks
2. **research** - For information gathering: research, analysis, data collection, comparative studies

IMPORTANT RULES:
- You are NOT a coding expert - DO NOT provide code solutions or technical implementations
- You are NOT a research assistant - DO NOT provide detailed analysis or research findings  
- You are ONLY a classifier - your response should ONLY contain the workflow decision
- NEVER write code, NEVER solve problems, NEVER provide detailed answers
- Your output MUST be valid JSON format ONLY - no additional text

CRITICAL: You MUST respond with ONLY a JSON object in this exact format:
{
    "workflow_type": "coding|research",
    "reasoning": "brief explanation of classification",
    "confidence": 0.0-1.0,
    "needs_additional_info": false
}

DO NOT add any other text, explanations, or answers before or after the JSON.
DO NOT use markdown formatting.
DO NOT include code examples.
"""

    async def analyze_request(self, user_input: str) -> OrchestratorDecision:
        """Analyze the request and make routing decision"""

        analysis_prompt = f"""
    USER REQUEST: {user_input}

    Classify this request and return ONLY JSON:
    """

        try:
            """
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=self.system_prompt),
                    SystemMessage(content=analysis_prompt),
                ]
            )
            """
            request = [
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
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                content = json_match.group()

            # Парсинг JSON
            decision_data = json.loads(content)

            # Убедимся, что все обязательные поля присутствуют
            if "needs_additional_info" not in decision_data:
                decision_data["needs_additional_info"] = False

            return OrchestratorDecision(**decision_data)

        except Exception as e:
            logger.error(f"{e}")
            # Fallback decision on error с ВСЕМИ обязательными полями
            return OrchestratorDecision(
                workflow_type="research",
                reasoning=f"Analysis error: {str(e)}",
                confidence=0.5,
                needs_additional_info=False,
            )

    async def execute_workflow(
        self, decision: OrchestratorDecision, user_input: str, context: Dict = None
    ) -> Any:
        """Execute the selected workflow"""

        if decision.workflow_type == "coding":
            if "coding" in self.workflows:
                from core.state import CodeWorkflowState

                state = CodeWorkflowState(user_input=user_input)
                return await self.workflows["coding"].run(state)
            else:
                return "Coding workflow is not registered"

        """
        elif decision.workflow_type == "research":
            if "research" in self.workflows:
                return await self.workflows["research"].run(user_input)
            else:
                return "Research workflow is not registered"
        

        else:
            return f"Unknown workflow type: {decision.workflow_type}"
        """

    async def _handle_direct_response(self, user_input: str) -> str:
        """Handle direct requests without specialized workflows"""

        direct_prompt = f"""
The user asked a question that doesn't require specialized workflow. 
Provide a direct, informative answer:

QUESTION: {user_input}

Answer clearly and to the point.
"""
        """
        response = await self.model.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=direct_prompt),
            ]
        )
        """

        request = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=direct_prompt),
        ]

        response_chunks = []

        async for chunk in self.model.astream(request):
            if hasattr(chunk, "content"):
                print(chunk.content, end="", flush=True)
                response_chunks.append(chunk.content)

        response = "".join(response_chunks)

        return response

    async def process_request(
        self, user_input: str, context: Dict = None
    ) -> Dict[str, Any]:
        """Main method for processing requests"""

        # Analyze request
        decision = await self.analyze_request(user_input)

        # Execute workflow
        result = await self.execute_workflow(decision, user_input, context)

        return {
            "decision": decision,
            "result": result,
            "workflow_used": decision.workflow_type,
        }
