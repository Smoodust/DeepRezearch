import traceback
from typing import List

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from loguru import logger

from core.state import (Code, CodeAnalysis, CodeReview, CodeWorkflowState,
                        WorkflowStep)

from .base_agent import BaseAgent
from .prompts import (CODE_GENERATION_PROMPT, CODE_GENERATION_TEMPLATE,
                      CODE_REVIEW_TEMPLATE, TASK_ANALYSIS_TEMPLATE)


class CodingAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
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
    
    @property
    def name(self):
        return "coding"

    @property
    def purpose(self):
        return "For ACTUAL programming tasks: writing code, debugging, code review, implementation, optimization, leetcode tasks."

    def _create_generation_agent(self):
        model = ChatOllama(
            model=self.model_name, format="json", temperature=0.1, num_predict=2048
        )
        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=CODE_GENERATION_PROMPT,
            response_format=ToolStrategy(Code),
        )

    async def analyze(self, task: str) -> CodeAnalysis:
        logger.debug(f"[{self.name}] 🔍 Starting analyse")

        analysis_prompt = TASK_ANALYSIS_TEMPLATE.format(task=task)

        try:
            response = await self.analysis_agent.ainvoke(analysis_prompt)

            logger.success(f"[{self.name}] ✅ Анализ завершен")
            logger.debug(f"[{self.name}] 📊 Результат анализа получен")

            return response
        except Exception as e:
            logger.error(
                f"[{self.name}] ❌ Ошибка в analyze(): {type(e).__name__}: {e}"
            )
            logger.error(f"[{self.name}] 📋 Трассировка:\n{traceback.format_exc()}")
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

        logger.debug(f"[{self.name}] 🔄 Начало генерации кода")

        try:
            agent_input = {"messages": [HumanMessage(content=generation_prompt)]}
            response = await self.generation_agent.ainvoke(agent_input)
            structured_response = response["structured_response"]

            logger.success(f"[{self.name}] ✅ Код сгенерирован успешно")

            return structured_response

        except Exception as e:
            logger.error(
                f"[{self.name}] ❌ Ошибка генерации кода: {type(e).__name__}: {e}"
            )
            raise RuntimeError(f"Failed to generate code: {str(e)}")

    async def review(self, code: Code, analysis: CodeAnalysis) -> CodeReview:
        review_prompt = CODE_REVIEW_TEMPLATE.format(
            code=code,
            requirements="\n".join(analysis.requirements),
            plan=analysis.plan.model_dump_json(),
        )

        logger.debug(f"[{self.name}] 🔄 Начало ревью кода")

        try:
            response = await self.review_agent.ainvoke(review_prompt)
            logger.success(f"[{self.name}] ✅ Ревью завершено")
            logger.debug(
                f"[{self.name}] 📊 Результат ревью: approved={response.approved}, качество={response.overall_quality}"
            )

            return response
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка ревью кода: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed to review code: {str(e)}")

    def build_graph(self) -> StateGraph:
        builder = StateGraph(CodeWorkflowState)

        builder.add_node("analyze", self._analyze_task)
        builder.add_node("generate_code", self._generate_code)
        builder.add_node("review_code", self._review_code)
        builder.add_node("reflect", self._reflect)
        builder.add_node("finalize", self._finalize)

        builder.set_entry_point("analyze")

        builder.add_edge("analyze", "generate_code")
        builder.add_edge("generate_code", "review_code")
        builder.add_conditional_edges(
            "review_code",
            self._should_reflect,
            {"reflect": "reflect", "finalize": "finalize"},
        )
        builder.add_edge("reflect", "generate_code")
        builder.add_edge("finalize", END)

        logger.success(f"[{self.name}] ✅ Workflow граф построен")
        return builder

    async def _analyze_task(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.ANALYSIS
            state.analysis = await self.analyze(state.user_input)
        except Exception as e:
            raise e
        return state

    async def _generate_code(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.GENERATION
            if state.analysis:
                feedback = state.metadata.get("feedback", [])
                state.generated_code = await self.generate(state.analysis, feedback)
            state.error = None
        except Exception as e:
            raise e
        return state

    async def _review_code(self, state: CodeWorkflowState) -> CodeWorkflowState:
        try:
            state.current_step = WorkflowStep.REVIEW
            print("review step")
            if state.analysis and state.generated_code:
                state.review = await self.review(state.generated_code, state.analysis)
        except Exception as e:
            raise e
        return state

    def _reflect(self, state: CodeWorkflowState) -> CodeWorkflowState:
        state.current_step = WorkflowStep.REFLECTION
        state.retry_count += 1

        print("reflect step")
        if state.review and not state.review.approved:
            state.metadata["feedback"] = state.review.suggestions
            state.needs_retry = True
        return state

    def _should_reflect(self, state: CodeWorkflowState) -> str:
        if state.review is None:
            return "finalize"

        if not state.review.approved and state.retry_count < state.max_retries:
            print("shoud reflect step")
            return "reflect"
        return "finalize"

    def _finalize(self, state: CodeWorkflowState) -> CodeWorkflowState:
        state.current_step = WorkflowStep.FINAL
        print("finalyze  step")
        if state.analysis and state.generated_code and state.review:
            state.final_result = {
                "analysis": state.analysis,
                "code": state.generated_code,
                "review": state.review,
            }
        return state
