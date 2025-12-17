import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

from agents.base_agent import BaseAgentState

############# RESEARCHER ################


class SearchQueriesStructureOutput(BaseModel):
    rationale: str
    query: list[str]


class RawDocument(TypedDict):
    url: str
    source: str


class SearchedDocument(TypedDict):
    url: str
    source: str
    extracted_info: str


class SearchedCollection(TypedDict):
    searched_documents: list[SearchedDocument]


class SearchWorkflowState(BaseAgentState):
    search_queries: list[str]
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
    steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step implementation plan. Provide at least 3 specific steps.",
    )
    libraries: List[str] = Field(
        default_factory=list,
        description="Required libraries ONLY from standard library. Use [] for standard library only.",
    )
    complexity: str = Field(
        default="Medium",
        description="Implementation complexity: 'Low', 'Medium', or 'High'.",
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Potential risks and challenges for this specific task.",
    )
    test_approach: Optional[List[str]] = Field(
        default=None, description="Testing strategy"
    )


class CodeAnalysis(BaseModel):
    task: str = Field(default="Unspecified task", description="Original user task")
    plan: CodePlan = Field(
        default_factory=CodePlan, description="Technical implementation plan"
    )
    requirements: List[str] = Field(
        default_factory=lambda: ["Functional requirements", "Performance requirements"],
        description="Functional requirements and acceptance criteria",
    )
    assumptions: List[str] = Field(
        default_factory=lambda: ["Standard environment", "User requirements are clear"],
        description="Any assumptions made during analysis",
    )


class Code(BaseModel):
    code: str = Field(description="Code written by agent")
    output: Optional[str] = Field(default=None, description="Actual execution output")


class LLMCodeReview(BaseModel):
    issues: List[str] = Field(description="list of identified problems")
    suggestions: List[str] = Field(description="list of improvements")
    security_concerns: List[str] = Field(
        default_factory=list,
        description="list of security issues",
    )
    overall_quality: int = Field(description="rating from 1-10", ge=1, le=10)

    @field_validator("issues", "suggestions", "security_concerns")
    def validate_list_contents(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class CodeReview(LLMCodeReview):
    approved: bool = Field(description="approved or not approved code")

    @classmethod
    def from_llm_review(
        cls, llm_review: LLMCodeReview, approved_threshold: int = 6
    ) -> "CodeReview":
        approved = llm_review.overall_quality >= approved_threshold
        return cls(
            issues=llm_review.issues,
            suggestions=llm_review.suggestions,
            security_concerns=llm_review.security_concerns,
            overall_quality=llm_review.overall_quality,
            approved=approved,
        )


class CodeAgentState(BaseAgentState):
    workflow_input: str
    current_step: str

    analysis_data: Optional[dict]
    generated_code_data: Optional[dict]
    review_data: Optional[dict]

    last_successful_generated_code: Optional[dict]

    retry_count: int

    errors: Annotated[list[str], operator.add]

    current_feedback: Annotated[list[str], operator.add]
    all_feedback: Annotated[list[str], operator.add]

    metadata: Dict[str, Any]
