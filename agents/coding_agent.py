from typing import List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

from core.state import CodeAnalysis, CodeReview, OverallCode

from .base_agent import BaseAgent
from .prompts import (CODE_GENERATION_PROMPT, CODE_GENERATION_TEMPLATE,
                      CODE_REVIEW_PROMPT, CODE_REVIEW_TEMPLATE,
                      CODING_ANALYSIS_PROMPT, TASK_ANALYSIS_TEMPLATE)


class CodingAgent(BaseAgent):
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = init_chat_model(model, model_provider="ollama")
        self.tools = [
            Tool(
                name="python_repl",
                description="Execute Python code. Input must be valid Python code. Use print() to see output.",
                func=PythonREPL().run,
            ),
        ]

        self.analysis_agent = self._create_analysis_agent()
        self.review_agent = self._create_review_agent()
        self.generation_agent = self._create_generation_agent()

    def _create_analysis_agent(self):
        structured_model = self.model.with_structured_output(CodeAnalysis)
        return create_agent(
            model=structured_model,
            tools=[],
            system_prompt=CODING_ANALYSIS_PROMPT,
            response_format=CodeAnalysis,
        )

    def _create_generation_agent(self):
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=CODE_GENERATION_PROMPT,
        )

    def _create_review_agent(self):
        structured_model = self.model.with_structured_output(CodeReview)

        return create_agent(
            model=structured_model,
            tools=self.tools,
            system_prompt=CODE_REVIEW_PROMPT,
            response_format=CodeReview,
        )

    async def analyze(self, task: str) -> CodeAnalysis:
        analysis_prompt = TASK_ANALYSIS_TEMPLATE.format(task=task)
        agent_input = {"messages": [HumanMessage(content=analysis_prompt)]}

        try:
            response = await self.analysis_agent.ainvoke(agent_input)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to analyze task: {str(e)}")

    async def generate(self, analysis: CodeAnalysis, feedback: List[str] = None) -> str:
        generation_prompt = CODE_GENERATION_TEMPLATE.format(
            task=analysis.task,
            plan=analysis.plan.model_dump_json(),
            requirements="\n".join(analysis.requirements),
            feedback="\n".join(feedback) if feedback else "No feedback",
        )

        try:
            response = await self.generation_agent.ainvoke(generation_prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to analyze task: {str(e)}")

    async def review(self, code: str, analysis: CodeAnalysis) -> CodeReview:
        review_prompt = CODE_REVIEW_TEMPLATE.format(
            code=code,
            requirements="\n".join(analysis.requirements),
            plan=analysis.plan.model_dump_json(),
        )

        try:
            response = await self.review_agent.ainvoke(review_prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to analyze task: {str(e)}")

    async def execute(self, context: str) -> OverallCode:
        analysis = await self.analyze(context)
        code = await self.generate(analysis)
        review = await self.review(code, analysis)

        return OverallCode(analysis=analysis, code=code, review=review)
