import traceback
from typing import List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from loguru import logger

from core.state import (Code, CodeAgentState, CodeAnalysis, CodeReview,
                        LLMCodeReview, WorkflowStep)

from .base_agent import BaseAgent, BaseAgentOutput
from .prompts import (CODE_GENERATION_PROMPT, CODE_GENERATION_TEMPLATE,
                      CODE_REVIEW_TEMPLATE, TASK_ANALYSIS_TEMPLATE)


class CodingAgent(BaseAgent):
    def __init__(
        self, model_name: str, max_retries: int = 3, approval_treshold: int = 6
    ):
        super().__init__()

        self.model_name = model_name
        self.max_retries = max_retries
        self.approval_treshold = approval_treshold

        self.model = init_chat_model(model_name, model_provider="ollama")
        self.tools = [
            Tool(
                name="python_repl",
                description="Execute Python code. Input must be valid Python code. Use print() to see output.",
                func=PythonREPL().run,
            ),
        ]

        self.analysis_agent = self.model.with_structured_output(CodeAnalysis)
        self.review_agent = self.model.with_structured_output(LLMCodeReview)
        self.generation_agent = self._create_generation_agent()

    @property
    def name(self):
        return "PYTHON_EXECUTOR - 'The Computational Engine'"

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

    def _create_generation_agent(self):
        model = ChatOllama(
            model=self.model_name, format="json", temperature=0.1, num_predict=2048
        )
        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=CODE_GENERATION_PROMPT,
        )

    async def analyze(self, task: str) -> CodeAnalysis:
        logger.debug(f"[{self.name}] 🔍 Starting analyse")

        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        analysis_prompt = TASK_ANALYSIS_TEMPLATE.format(task=task)

        try:
            response: CodeAnalysis = await self.analysis_agent.ainvoke(analysis_prompt)
            logger.debug(f"Raw response from LLM: {response}")

            if not response or not isinstance(response, CodeAnalysis):
                raise ValueError("Invalid analysis response format")

            if not response.task:
                response.task = task

            logger.success(f"[{self.name}] ✅ Analysis completed")
            logger.debug(f"[{self.name}] 📊 Analysis result received")

            return response
        except Exception as e:
            logger.error(
                f"[{self.name}] ❌ Error in analyze(): {type(e).__name__}: {e}"
            )
            logger.error(f"[{self.name}] Trace:\n{traceback.format_exc()}")

            return CodeAnalysis(
                task=task,
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
        self, analysis: CodeAnalysis, feedback: List[str] = None
    ) -> Code:
        if not analysis:
            raise ValueError("Analysis cannot be None")

        generation_prompt = CODE_GENERATION_TEMPLATE.format(
            task=analysis.task,
            plan=analysis.plan.model_dump_json(),
            requirements="\n".join(analysis.requirements),
            feedback="\n".join(feedback[-3:] if feedback else ["No feedback"]),
        )

        logger.debug(f"[{self.name}] 🔄 Start generating code")

        try:
            agent_input = {"messages": [HumanMessage(content=generation_prompt)]}
            response: dict = await self.generation_agent.ainvoke(agent_input)

            messages = response["messages"]

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

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Critical generation failure: {str(e)}")
            logger.debug(f"[{self.name}] Input prompt:\n{generation_prompt}")

            if "code_str" not in locals():
                code_str = "[Code generation failed]"

            if "recursion" in str(e).lower() or "GRAPH_RECURSION_LIMIT" in str(e):
                logger.error(
                    f"[{self.name}] ❌ Agent stuck in infinite loop. Check agent configuration or task complexity."
                )
                print(code_str)
                return Code(
                    code=code_str,
                    output=f"Agent failed due to recursion limit. Try simplifying the task or increasing recursion_limit. Error: {str(e)}",
                )

            return Code(code=code_str, output=str(e))

    async def review(self, code: Code, analysis: CodeAnalysis) -> CodeReview:
        if not code or not code.code:
            raise ValueError("Code cannot be empty for review")

        if not analysis:
            raise ValueError("Analysis cannot be None for review")

        review_prompt = CODE_REVIEW_TEMPLATE.format(
            code=code.model_dump(),
            requirements="\n".join(analysis.requirements),
            plan=analysis.plan.model_dump_json(),
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

            generated_code = await self.generate(analysis, feedback)

            result_state["generated_code_data"] = generated_code.model_dump()

            try:
                code_text = (generated_code.code or "").strip()
                if code_text and not code_text.startswith("# Code generation failed"):
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

            review = await self.review(generated_code, analysis)

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
                code = Code.model_validate(state["generated_code_data"])
                final_result = f"Code: {code.code}"

                if code.output:
                    final_result += f"\nOutput: {code.output}"
            except Exception as e:
                final_result = f"Error finalizing output: {str(e)}"

        return {"output": final_result}
