import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

from agents.base_agent import BaseAgentState

from ..agents.base_agent import BaseAgentState

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
        return v


class CodeAgentState(BaseAgentState):
    workflow_input: str
    current_step: str

    analysis_data: Optional[dict]
    generated_code_data: Optional[dict]
    review_data: Optional[dict]

    needs_retry: bool
    retry_count: int
    max_retries: int

    errors: Annotated[list[str], operator.add]
    feedback: Annotated[list[str], operator.add]

    metadata: Dict[str, Any]
