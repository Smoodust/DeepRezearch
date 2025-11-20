from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field, validator
from datetime import datetime
from langgraph.graph import add_messages
from enum import Enum
from dataclasses import dataclass, field
import operator


class OverallState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    search_query: List[str] = Field(default_factory=list)
    web_research_result: List[Any] = Field(default_factory=list)
    sources_gathered: List[str] = Field(default_factory=list)
    initial_search_query_count: int = 3
    max_research_loops: int = 5
    research_loop_count: int = 0
    #reasoning_model: str 


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: List[str]
    research_loop_count: int
    number_of_ran_queries: int


class Query(BaseModel):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: List[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)