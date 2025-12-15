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

from core.state import (Code, CodeAgentState, CodeAnalysis, CodeReview,
                        GeneratedCode, WorkflowStep)

from .base_agent import BaseAgent, BaseAgentOutput
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
            response_format=ToolStrategy(GeneratedCode),
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
            print(response)
            code_str = response["structured_response"].code

            logger.debug(f"[{self.name}] ⚙️ Executing:\n{code_str}")
            try:
                output = self.tools[0].func(code_str)
            except Exception as e:
                output = f"EXECUTION FAILED: {str(e)}"
                logger.warning(f"[{self.name}] ⚠️ Execution error captured")

            full_code = Code(
                code=code_str,
                output=output.strip() if output else "[No output produced]",
            )
            logger.success(f"[{self.name}] ✅ Generated and executed code")
            return full_code

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Critical generation failure: {str(e)}")
            logger.debug(f"[{self.name}] Input prompt:\n{generation_prompt}")
            raise RuntimeError(f"Code generation pipeline failed: {str(e)}")

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
        builder = StateGraph(
            CodeAgentState,
            output_schema=BaseAgentOutput,
        )

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

    async def _analyze_task(self, state: CodeAgentState) -> CodeAgentState:
        try:
            result_state = dict(state)
            result_state["current_step"] = WorkflowStep.ANALYSIS.value
            analysis = await self.analyze(state["workflow_input"])

            result_state["analysis_data"] = analysis.model_dump()
            result_state["errors"] = []

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            result_state = dict(state)
            result_state["current_step"] = WorkflowStep.ANALYSIS.value
            result_state["errors"] = [f"Analysis failed: {str(e)}"]
            return result_state

        return result_state

    async def _generate_code(self, state: CodeAgentState) -> CodeAgentState:
        try:
            result_state = dict(state)
            result_state["current_step"] = WorkflowStep.GENERATION.value

            if not state.get("analysis_data"):
                raise ValueError("No analysis data available")

            analysis = CodeAnalysis.model_validate(state["analysis_data"])
            feedback = state.get("feedback", [])
            generated_code = await self.generate(analysis, feedback)

            result_state["generated_code_data"] = generated_code.model_dump()
            result_state["errors"] = []

            return result_state
        except Exception as e:
            logger.error(f"Generation error: {e}")
            result_state = dict(state)
            result_state["current_step"] = WorkflowStep.GENERATION.value
            result_state["errors"] = state.get("errors", []) + [
                f"Generation failed: {str(e)}"
            ]
            return result_state

    async def _review_code(self, state: CodeAgentState) -> CodeAgentState:
        try:
            result_state = dict(state)
            result_state["current_step"] = WorkflowStep.REVIEW.value

            if not state.get("analysis_data") or not state.get("generated_code_data"):
                logger.warning("Missing analysis or generated code data")
                return result_state

            analysis = CodeAnalysis.model_validate(state["analysis_data"])
            generated_code = Code.model_validate(state["generated_code_data"])

            review = await self.review(generated_code, analysis)

            result_state["review_data"] = review.model_dump()
            result_state["feedback"] = review.suggestions if not review.approved else []

            return result_state
        except Exception as e:
            logger.error(f"Review error: {e}")
            return state

    async def _reflect(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.REFLECTION.value
        result_state["retry_count"] = state.get("retry_count", 0) + 1
        result_state["needs_retry"] = True
        return result_state

    def _should_reflect(self, state: CodeAgentState) -> str:
        review_data = state.get("review_data")

        if not review_data:
            return "finalize"

        review = CodeReview.model_validate(review_data)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if not review.approved and retry_count < max_retries:
            return "reflect"
        return "finalize"

    async def _finalize(self, state: CodeAgentState) -> BaseAgentOutput:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.FINAL.value

        final_result = ""
        if state.get("generated_code_data"):
            code_data = state["generated_code_data"]
            code = Code.model_validate(code_data)
            final_result = f"Code: {code.code}"
            if code.output:
                final_result += f"\nOutput: {code.output}"

        return {"output": final_result}
