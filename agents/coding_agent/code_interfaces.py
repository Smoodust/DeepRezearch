from typing import Optional, Protocol

from core.state import Code, CodeAnalysis, CodeReview


class ICodeAnalyzer(Protocol):
    async def analyze(self, task: str) -> CodeAnalysis:
        """Analyze a coding task and return structured analysis."""
        ...


class ICodeGenerator(Protocol):
    async def generate(
        self, task: str, analysis: CodeAnalysis, feedback: Optional[list[str]] = None
    ) -> Code:
        """Generate code based on task and analysis."""
        ...


class ICodeReviewer(Protocol):
    async def review(self, task: str, code: Code, analysis: CodeAnalysis) -> CodeReview:
        """Review generated code and provide feedback."""
        ...
