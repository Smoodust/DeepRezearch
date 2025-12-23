import time
import traceback
import uuid
from enum import Enum
from typing import List, Literal, Optional, Type

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from core.state import (Code, CodeAgentState, CodeAnalysis, CodeReview,
                        LLMCodeReview, WorkflowStep)

from .base_agent import BaseAgent, BaseAgentOutput, BaseAgentState


class IntentEnum(str, Enum):
    """Possible intents of a user request."""

    CODE_GENERATION = "code_generation"
    СALCULATION = "make_calculations"
    OTHER = "other"


class ComplexityEnum(str, Enum):
    """Estimated complexity of the task."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CodingUserInput(BaseModel):
    # The fields are defined in the order the LLM should reason through them.
    workflow_type: Literal["PYTHON_EXECUTOR"]

    step1_summary: str = Field(description="A concise summary of the user's request.")
    step2_intent: IntentEnum = Field(description="The primary intent of the request.")
    step3_complexity: ComplexityEnum = Field(
        description="Estimated complexity of the task."
    )
    step4_required_actions: List[str] = Field(
        description="Specific actions the coder agent should perform (e.g., 'write a function', 'debug', 'explain code')."
    )
    step5_potential_pitfalls: Optional[List[str]] = Field(
        default=None,
        description="Potential pitfalls or challenges the coder agent should be aware of.",
    )
    step6_test_required: bool = Field(
        default=False, description="Whether the task likely requires writing tests."
    )
    step7_security_considerations: Optional[List[str]] = Field(
        default=None,
        description="Security-related considerations (e.g., unsafe functions, input validation).",
    )
    selected_context_ids: List[int] = Field(
        default_factory=list,
        description="List of ids from context that should be given to agent",
    )


class CodingAgent(BaseAgent):
    def __init__(
        self, model_name: str, max_retries: int = 3, approval_treshold: int = 6
    ):
        super().__init__()

        self.model_name = model_name

        self.checkpointer = InMemorySaver()
        self.thread_id = None

        self.max_retries = max_retries
        self.approval_treshold = approval_treshold

        self._analysis_tpl = None
        self._generation_tpl = None
        self._review_tpl = None
        self._system_tpl = None

        self.model = init_chat_model(model_name, model_provider="ollama")

        self.tools = [PythonREPLTool()]

        self.analysis_agent = self.model.with_structured_output(CodeAnalysis)
        self.review_agent = self.model.with_structured_output(LLMCodeReview)
        self.generation_agent = self._create_generation_agent()

    @property
    def name(self):
        return "PYTHON_EXECUTOR"

    @property
    def purpose(self):
        purpose = """
        Write and Execute Python code for calculations, data processing, and algorithmic tasks
        Capabilities:
        - Python REPL environment with standard libraries only
        - Mathematical computations and data transformations
        - Script execution for well-defined programming tasks
        - No internet access or external libraries
        Use when: Task requires computation, data manipulation, or algorithmic processing
        """
        return purpose

    @property
    def get_input_model(self) -> Type[BaseModel]:
        return CodingUserInput

    def _create_generation_agent(self):
        model = ChatOllama(
            model=self.model_name, format="json", temperature=0.1, num_predict=2048
        )

        system_prompt = self._load_template(
            "coding_agent/CODE_GENERATION_PROMPT.jinja"
        ).render()

        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
        )

    async def analyze(self, task: str) -> CodeAnalysis:
        logger.debug(f"[{self.name}] 🔍 Starting analyse")

        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        if self._analysis_tpl is None:
            self._analysis_tpl = self._load_template(
                "coding_agent/TASK_ANALYSIS_TEMPLATE.jinja"
            )

        analysis_prompt = self._analysis_tpl.render(task=task)

        try:
            response: CodeAnalysis = await self.analysis_agent.ainvoke(analysis_prompt)
            logger.debug(f"Raw response from LLM: {response}")

            if not response or not isinstance(response, CodeAnalysis):
                raise ValueError("Invalid analysis response format")

            logger.success(f"[{self.name}] ✅ Analysis completed")
            logger.debug(f"[{self.name}] 📊 Analysis result received")

            return response
        except Exception as e:
            logger.error(
                f"[{self.name}] ❌ Error in analyze(): {type(e).__name__}: {e}"
            )
            logger.error(f"[{self.name}] Trace:\n{traceback.format_exc()}")

            return CodeAnalysis(
                steps=[
                    "Define implementation approach",
                    "Write code",
                    "Test functionality",
                ],
                libraries=[],
                complexity="Medium",
                risks=["Time constraints", "Technical dependencies"],
                test_approach=None,
            )

    async def generate(
        self, task: str, analysis: CodeAnalysis, feedback: List[str] = None
    ) -> Code:
        if not analysis:
            raise ValueError("Analysis cannot be None")

        if self._generation_tpl is None:
            self._generation_tpl = self._load_template(
                "coding_agent/CODE_GENERATION_TEMPLATE.jinja"
            )

        generation_prompt = self._generation_tpl.render(
            task=task,
            plan=analysis.model_dump(),
            requirements=analysis.requirements,
            feedback=feedback[-3:] if feedback else [],
        )

        logger.debug(f"[{self.name}] 🔄 Start generating code")

        messages = []

        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            agent_input = {"messages": [HumanMessage(content=generation_prompt)]}
            response: dict = await self.generation_agent.ainvoke(
                agent_input, config=config
            )

            messages = response.get("messages", [])

            tool_output = next(
                (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
                "[No output produced]",
            )

            ai_messsage = next(
                (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
                "[No output produced]",
            )

            code_str = ai_messsage

            logger.debug(f"{code_str}\n{tool_output}")

            logger.success(f"[{self.name}] ✅ Generated and executed code")

            return Code(code=code_str, output=tool_output)

        except GraphRecursionError:
            logger.error(
                f"[{self.name}] ❌ Agent stuck in infinite loop. Check agent configuration or task complexity."
            )

            state = self.generation_agent.get_state(config)
            result = state.values
            messages = result.get("messages", [])

            tool_output = next(
                (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
                "[No output produced]",
            )

            ai_messsage = next(
                (m.tool_calls for m in reversed(messages) if isinstance(m, AIMessage)),
                "[No output produced]",
            )

            for tc in ai_messsage:
                if tc.get("name") == "python_repl":
                    args = tc.get("args", {})
                    if "__arg1" in args:
                        code_str = args["__arg1"]

            logger.debug(f"{code_str}\n{tool_output}")

            return Code(
                code=code_str,
                output=tool_output,
            )

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Critical generation failure: {str(e)}")
            logger.debug(f"[{self.name}] Input prompt:\n{generation_prompt}")

            if "code_str" not in locals():
                code_str = "[Code generation failed]"

            return Code(code=code_str, output=str(e))

    async def review(self, task: str, code: Code, analysis: CodeAnalysis) -> CodeReview:
        if not code or not code.code:
            raise ValueError("Code cannot be empty for review")

        if not analysis:
            raise ValueError("Analysis cannot be None for review")

        if self._review_tpl is None:
            self._review_tpl = self._load_template(
                "coding_agent/CODE_REVIEW_TEMPLATE.jinja"
            )

        review_prompt = self._review_tpl.render(
            task=task,
            code=code.model_dump(),
            requirements=analysis.requirements,
            plan=analysis.model_dump(),
        )

        logger.debug(f"[{self.name}] 🔄 Start reviewing code")

        try:
            response: LLMCodeReview = await self.review_agent.ainvoke(review_prompt)

            if not isinstance(response, LLMCodeReview):
                raise ValueError("Invalid review response format")

            code_review = CodeReview.from_llm_review(
                response, approved_threshold=self.approval_treshold
            )

            logger.success(f"[{self.name}] ✅ Review completed")
            logger.debug(
                f"[{self.name}] 📊 Review result: approved={code_review.approved}, "
                f"quality={code_review.overall_quality}"
            )

            return code_review
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error in review(): {type(e).__name__}: {e}")

            return CodeReview(
                approved=False,
                overall_quality=0,
                suggestions=[
                    "Review failed due to technical error. Please check the code manually."
                ],
            )

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

        self.thread_id = f"{uid}-{ts}"

        return await self.compiled_graph.ainvoke(state)  # type: ignore

    async def _analyze_task(self, state: CodeAgentState) -> CodeAgentState:
        result_state = dict(state)
        result_state["current_step"] = WorkflowStep.ANALYSIS.value

        try:
            analysis = await self.analyze(state["workflow_input"])

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

            generated_code = await self.generate(
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

            review = await self.review(
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

        try:
            await self.checkpointer.adelete_thread(self.thread_id)
        except Exception as e:
            logger.warning(
                "[%s] Failed to cleanup checkpointer for thread_id=%s: %s",
                self.name,
                self.thread_id,
                e,
            )
        finally:
            self.thread_id = None

        return {"output": final_result}
