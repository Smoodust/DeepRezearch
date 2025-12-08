import re
import time
from typing import Optional, cast

import requests
from base_agent import BaseAgent
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from prompts import SITE_INFO_EXTRACTION_TEMPALTE

from core.state import RawDocument, SearchedDocument

options = ConversionOptions()
options.extract_metadata = False
options.autolinks = False


class ResearchAgent(BaseAgent):
    def __init__(self, name: str, model_name: str, max_result: int):
        self.name = name
        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
        self.max_result = max_result
        self._compiled_graph = None

    def searching(self, query: str) -> list[RawDocument]:
        search_results = DDGS().text(query, max_results=self.max_result)  # type: ignore
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
        return results

    async def extract_text_from_search(self, doc: RawDocument) -> SearchedDocument:
        markdown = convert(doc["source"], options)
        markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)

        context = SITE_INFO_EXTRACTION_TEMPALTE.format(markdown=markdown)
        response = await self.model.ainvoke(context)
        return {**doc, "extracted_info": cast(str, response.content)}

    @abstractmethod
    async def execute(self, context: str) -> Any:
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass

    @property
    def compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    async def run(self, state):
        return await self.compiled_graph.ainvoke(state)
