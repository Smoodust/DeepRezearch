import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

############# RESEARCHER ################


class RawDocument(TypedDict):
    url: str
    source: str


class SearchedDocument(TypedDict):
    url: str
    source: str
    extracted_info: str


class SearchedCollection(TypedDict):
    searched_documents: list[SearchedDocument]


class SearchWorkflowState(TypedDict):
    search_query: str
    sources: list[RawDocument]
    searched_documents: Annotated[list[SearchedDocument], operator.add]


############# CODER ################
class WorkflowStep(str, Enum):
    INIT = "init"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    REVIEW = "review"
    REFLECTION = "reflection"
    FINAL = "final"


class CodePlan(BaseModel):
    steps: List[str] = Field(description="Step-by-step implementation plan")
    libraries: List[str] = Field(description="Required libraries and dependencies")
    complexity: str = Field(description="Implementation complexity level")
    risks: List[str] = Field(description="Potential risks and challenges")
    test_approach: Optional[List[str]] = Field(description="Testing strategy")


class CodeAnalysis(BaseModel):
    task: str = Field(description="Original user task")
    plan: CodePlan = Field(description="Technical implementation plan")
    requirements: List[str] = Field(
        description="Functional requirements and acceptance criteria"
    )
    assumptions: List[str] = Field(description="Any assumptions made during analysis")


class GeneratedCode(BaseModel):
    code: str = Field(description="Complete Python code")


class Code(BaseModel):
    code: str = Field(description="Code written by agent")
    output: Optional[str] = Field(default=None, description="Actual execution output")


class CodeReview(BaseModel):
    approved: bool = Field(description="approved or not approved code")
    issues: List[str] = Field(description="list of identified problems")
    suggestions: List[str] = Field(description="list of improvements")
    security_concerns: List[str] = Field(description="list of security issues")
    overall_quality: int = Field(description="rating from 1-10", ge=1, le=10)

    @field_validator("issues", "suggestions", "security_concerns")
    def validate_list_contents(cls, v):
        if v is None:
            return []
        return


"""
class OverallCode(BaseModel):
    analysis: CodeAnalysis
    code: Code
    review: CodeReview
"""


class CodeWorkflowState(BaseModel):
    user_input: str = Field(description="Original user request")

    # Current execution state
    current_step: WorkflowStep = Field(default=WorkflowStep.INIT)
    error: Optional[str] = Field(default=None, description="Any error that occurred")

    # Workflow data
    analysis: Optional[CodeAnalysis] = Field(default=None)
    generated_code: Optional[Code] = Field(default=None)
    review: Optional[CodeReview] = Field(default=None)
    final_result: str = Field(default=None)

    # Control flow
    needs_retry: bool = Field(default=False)
    retry_count: int = Field(default=3)
    max_retries: int = Field(default=3)

    # Metadata
    # execution_time: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
