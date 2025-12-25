import time
import uuid
from typing import Optional, Type

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel

from core.config import MODEL_URL

from ..base_agent import BaseAgent, BaseAgentOutput, BaseAgentState
from .code_analyzer import CodeAnalyzer
from .code_generator import CodeGenerator
from .code_interfaces import ICodeAnalyzer, ICodeGenerator, ICodeReviewer
from .code_reviewer import CodeReviewer
from .code_state import (Code, CodeAgentState, CodeAnalysis, CodeReview,
                         WorkflowStep)
from .coder_input import CodingUserInput


class CodingAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        purpose: str,
        model_name: str,
        analyzer: Optional[ICodeAnalyzer],
        generator: Optional[ICodeGenerator],
        reviewer: Optional[ICodeReviewer],
        chat: type[BaseChatModel] = ChatOllama,
        max_retries: int = 3,
        approval_threshold: int = 6,
        max_feedback_items: int = 3,
        max_stored_feedback: int = 10,
        temperature: float = 0.1,
        num_predict: int = 2048,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        super().__init__()

        model = init_chat_model(model_name, model_provider="ollama", base_url=MODEL_URL)
        checkpointer = checkpointer
        tools = [PythonREPLTool()]  # default tool

        self.max_retries = max_retries

        self.analyzer = analyzer or CodeAnalyzer(model)
        self.generator = generator or CodeGenerator(chat, model_name, temperature, num_predict, tools, checkpointer=checkpointer)  # type: ignore
        self.reviewer = reviewer or CodeReviewer(model, approval_threshold)

        self._name = name
        self._purpose = purpose

    @property
    def name(self):
        return self._name

    @property
    def purpose(self):
        return self._purpose

    @property
    def get_input_model(self) -> Type[BaseModel]:
        return CodingUserInput

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

        logger.success(
            f"[{self.name}] ✅ The workflow graph has been successfully built"
        )
        return builder

    async def run(self, state: BaseAgentState) -> BaseAgentOutput:
        uid = uuid.uuid4().hex[:12]
        ts = int(time.time() * 1000)

        self.generator.set_thread_id(f"{uid}-{ts}")

        return await self.compiled_graph.ainvoke(state)  # type: ignore

    async def _analyze_task(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.ANALYSIS.value

        try:
            analysis = await self.analyzer.analyze(state["workflow_input"])

            result_state["analysis_data"] = analysis.model_dump()
            result_state["errors"] = []

        except Exception as e:
            logger.error(f"Analysis error: {e}")

            result_state["errors"] = [f"Analysis failed: {str(e)}"]
            result_state["analysis_data"] = CodeAnalysis(
                task=state["workflow_input"],
                requirements=["Task analysis failed"],
                assumptions=["Fallback mode"],
            ).model_dump()

            return result_state

        return result_state

    async def _generate_code(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.GENERATION.value

        try:
            if not state.get("analysis_data"):
                raise ValueError("No analysis data available")

            analysis = CodeAnalysis.model_validate(state["analysis_data"])
            feedback = state.get("current_feedback", [])

            retry_count = state.get("retry_count", 0)

            logger.info(f"Try №{retry_count + 1} out of {self.max_retries}")

            generated_code = await self.generator.generate(
                state["workflow_input"], analysis, feedback
            )

            result_state["generated_code_data"] = generated_code.model_dump()

            try:
                code_text = (generated_code.code or "").strip()
                if code_text and not code_text.startswith("[Code generation failed]"):
                    result_state["last_successful_generated_code"] = (
                        generated_code.model_dump()
                    )
            except Exception:
                logger.debug(
                    "Failed to store last_successful_generated_code", exc_info=True
                )

            result_state["errors"] = []

            return result_state
        except Exception as e:
            logger.error(f"Generation error: {e}")

            result_state["errors"] = state.get("errors", []) + [
                f"Generation failed: {str(e)}"
            ]
            result_state["generated_code_data"] = Code(
                code="# Code generation failed",
                output=f"Error: {str(e)}",
                metadata={"generation_error": True},
            ).model_dump()

            return result_state

    async def _review_code(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.REVIEW.value

        try:
            if not state.get("analysis_data") or not state.get("generated_code_data"):
                logger.warning("Missing analysis or generated code data")
                result_state["review_data"] = CodeReview(
                    approved=False,
                    overall_quality=0,
                    suggestions=["Missing data for review"],
                ).model_dump()
                return result_state

            analysis = CodeAnalysis.model_validate(state["analysis_data"])
            generated_code = Code.model_validate(state["generated_code_data"])

            review = await self.reviewer.review(
                result_state["workflow_input"], generated_code, analysis
            )

            result_state["review_data"] = review.model_dump()
            result_state["feedback"] = review.suggestions if not review.approved else []

            return result_state

        except Exception as e:
            error_msg = f"Review error: {str(e)}"
            logger.error(error_msg)

            result_state["errors"] = state.get("errors", []) + [error_msg]
            result_state["has_error"] = True
            result_state["review_data"] = CodeReview(
                approved=False,
                overall_quality=0,
                suggestions=[f"Review failed: {str(e)}"],
            ).model_dump()

            return result_state

    async def _reflect(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.REFLECTION.value
        result_state["retry_count"] = state.get("retry_count", 0) + 1

        try:
            if state.get("review_data"):
                review = CodeReview.model_validate(state["review_data"])
                suggestions = review.suggestions[:3] if review.suggestions else []
            else:
                suggestions = ["No review data available"]

            result_state["current_feedback"] = suggestions

            all_feedback = state.get("all_feedback", [])

            for suggestion in suggestions:
                if suggestion not in all_feedback:
                    all_feedback.append(suggestion)

            if len(all_feedback) > 10:
                all_feedback = all_feedback[-10:]

            result_state["all_feedback"] = all_feedback

        except Exception as e:
            logger.error(f"Reflection error: {e}")
            result_state["current_feedback"] = ["Reflection failed"]

        return result_state

    def _should_reflect(self, state: CodeAgentState) -> str:
        review_data = state.get("review_data")

        if not review_data:
            return "finalize"

        try:
            review = CodeReview.model_validate(review_data)
            retry_count = state.get("retry_count", 0)

            if retry_count >= self.max_retries or review.approved:
                return "finalize"

            return "reflect"
        except Exception:
            return "finalize"

    async def _finalize(self, state: CodeAgentState) -> BaseAgentOutput:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.FINAL.value

        final_result = ""
        if state.get("generated_code_data"):
            try:
                succesfull_generation = (
                    state["generated_code_data"]
                    if state["review_data"]["approved"]
                    else state["last_successful_generated_code"]
                )
                code = Code.model_validate(succesfull_generation)
                final_result = f"Code: {code.code}"

                if code.output:
                    final_result += f"\nOutput: {code.output}"
            except Exception as e:
                final_result = f"Error finalizing output: {str(e)}"

        logger.success(f"FINAL: { {"output": final_result} }")

        await self.generator.delete_thread_id()

        return {"output": final_result}
