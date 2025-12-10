import re
import time
from typing import Optional, cast

import requests
from base_agent import BaseAgent
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from prompts import SITE_INFO_EXTRACTION_TEMPALTE

from core.state import RawDocument, SearchedDocument, SearchWorkflowState

options = ConversionOptions()
options.extract_metadata = False
options.autolinks = False


class ResearchAgent(BaseAgent):
    def __init__(self, model_name: str, max_result: int):
        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
        self.max_result = max_result
        self._compiled_graph = None
    
    @property
    def name(self):
        return "research"

    @property
    def purpose(self):
        return "For information gathering: research, analysis, data collection, comparative studies."

    def searching(self, state: SearchWorkflowState) -> SearchWorkflowState:
        search_results = DDGS().text(state["search_query"], max_results=self.max_result)  # type: ignore
        results = []
        for x in search_results:
            time.sleep(0.5)
            url: Optional[str] = x.get("href", None)
            if url is None:
                continue
            r = requests.get(
                url,
                headers={
                    "User-Agent": "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
                },
            )
            document: RawDocument = {"url": url, "source": r.text}
            results.append(document)
        state["sources"] = results
        return state

    async def extract_text_from_search(self, state: RawDocument):
        markdown = convert(doc["source"], options)
        markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)

        context = SITE_INFO_EXTRACTION_TEMPALTE.format(markdown=markdown)
        response = await self.model.ainvoke(context)
        document: SearchedDocument = {
            "url": state["url"],
            "source": state["source"], 
            "extracted_info": cast(str, response.content)
        }
        return {"searched_documents": [document]}
    
    def search_map(self, state: SearchWorkflowState):
        return [Send("extract_info", d) for d in state["sources"]]
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        builder = StateGraph(SearchWorkflowState)

        builder.add_node("searching", self.searching)
        builder.add_node("extract_info", self.extract_text_from_search)

        builder.set_entry_point("searching")

        builder.add_conditional_edges("searching", self.search_map, ["extract_info"])
        builder.add_edge("reflect", END)

        logger.success(f"[{self.name}] ✅ Workflow граф построен")
        return builder

    @property
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    async def run(self, state):
        return await self.compiled_graph.ainvoke(state)
