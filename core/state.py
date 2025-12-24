import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel

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


