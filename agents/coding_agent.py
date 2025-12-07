from typing import List

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama

from core.state import Code, CodeAnalysis, CodeReview, OverallCode

from .base_agent import BaseAgent
from .prompts import (CODE_GENERATION_PROMPT, CODE_GENERATION_TEMPLATE,
                      CODE_REVIEW_TEMPLATE, TASK_ANALYSIS_TEMPLATE)


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

        self.analysis_agent = self.model.with_structured_output(CodeAnalysis)
        self.review_agent = self.model.with_structured_output(CodeReview)
        self.generation_agent = self._create_generation_agent()

    def _create_generation_agent(self):
        model = ChatOllama(
            model="llama3.1:8b", format="json", temperature=0.1, num_predict=2048
        )
        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=CODE_GENERATION_PROMPT,
            response_format=ToolStrategy(Code),
        )

    async def analyze(self, task: str) -> CodeAnalysis:
        import asyncio
        import time

        print(f"\n🔍 НАЧАЛО АНАЛИЗА ЗАДАЧИ:")
        print(f"📋 Задача: {task}")

        analysis_prompt = TASK_ANALYSIS_TEMPLATE.format(task=task)
        print(f"📝 Длина промпта: {len(analysis_prompt)} символов")
        print(f"📄 Промпт (первые 500 символов): {analysis_prompt[:500]}...")

        try:
            print("🔄 Вызов analysis_agent.ainvoke()...")
            start_time = time.time()

            response = await asyncio.wait_for(
                self.analysis_agent.ainvoke(analysis_prompt), timeout=120.0
            )

            end_time = time.time()
            print(f"✅ Анализ завершен за {end_time - start_time:.2f} секунд")
            print(f"📊 Результат анализа: {response}")

            return response
        except asyncio.TimeoutError:
            print("❌ ТАЙМАУТ: Анализ занял более 60 секунд")
            raise RuntimeError("Analysis timeout - модель не ответила вовремя")
        except Exception as e:
            print(f"❌ Ошибка в analyze(): {type(e).__name__}: {e}")
            import traceback

            print(f"📋 Трассировка:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to analyze task: {str(e)}")

    async def generate(
        self, analysis: CodeAnalysis, feedback: List[str] = None
    ) -> Code:
        generation_prompt = CODE_GENERATION_TEMPLATE.format(
            task=analysis.task,
            plan=analysis.plan.model_dump_json(),
            requirements="\n".join(analysis.requirements),
            feedback="\n".join(feedback) if feedback else "No feedback",
        )

        try:
            agent_input = {"messages": [HumanMessage(content=generation_prompt)]}
            print("🔄 generation model await for response")
            response = await self.generation_agent.ainvoke(agent_input)

            structured_response = response["structured_response"]
            print(f"✅ Generated code: {structured_response}")

            return response["structured_response"]

        except Exception as e:
            raise RuntimeError(f"Failed to generate code: {str(e)}")

    async def review(self, code: Code, analysis: CodeAnalysis) -> CodeReview:
        review_prompt = CODE_REVIEW_TEMPLATE.format(
            code=code,
            requirements="\n".join(analysis.requirements),
            plan=analysis.plan.model_dump_json(),
        )

        try:
            print("review model await for response")
            response = await self.review_agent.ainvoke(review_prompt)
            print(response)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to review code: {str(e)}")

    async def execute(self, context: str) -> OverallCode:
        analysis = await self.analyze(context)
        code = await self.generate(analysis)
        review = await self.review(code, analysis)

        return OverallCode(analysis=analysis, code=code, review=review)
