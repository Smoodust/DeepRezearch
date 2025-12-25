from langchain_core.language_models import BaseChatModel
from loguru import logger

from core.template_manager import TemplateManager

from .code_interfaces import ICodeAnalyzer
from .code_state import CodeAnalysis


class CodeAnalyzer(ICodeAnalyzer):
    def __init__(self, model: BaseChatModel):
        self.analysis_agent = model.with_structured_output(CodeAnalysis)
        self.name = "CodeAnalyzer"

    async def analyze(self, task: str) -> CodeAnalysis:
        logger.debug(f"[{self.name}] 🔍 Starting analyse")

        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        analysis_prompt = TemplateManager().render_template(
            "coding_agent/TASK_ANALYSIS_TEMPLATE.jinja", task=task
        )

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

            return self._create_fallback_analysis(task)

    def _create_fallback_analysis(self, task: str) -> CodeAnalysis:
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
