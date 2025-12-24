from langchain_core.language_models import BaseChatModel
from loguru import logger

from .code_state import Code, CodeAnalysis, CodeReview, LLMCodeReview
from core.template_manager import TemplateManager

from .code_config import CodingAgentConfig


class CodeReviewer:
    def __init__(self, model: BaseChatModel, config: CodingAgentConfig):
        self.review_agent = model.with_structured_output(LLMCodeReview)
        self.name = "CodeReviewer"

        self.approval_treshold = config.approval_threshold

    async def review(self, task: str, code: Code, analysis: CodeAnalysis) -> CodeReview:
        if not code or not code.code:
            raise ValueError("Code cannot be empty for review")

        if not analysis:
            raise ValueError("Analysis cannot be None for review")

        review_prompt = TemplateManager().render_template(
            "coding_agent/CODE_REVIEW_TEMPLATE.jinja",
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

            return self._create_fallback_review()

    def _create_fallback_review(self) -> CodeReview:
        return CodeReview(
            approved=False,
            overall_quality=0,
            suggestions=[
                "Review failed due to technical error. Please check the code manually."
            ],
        )
