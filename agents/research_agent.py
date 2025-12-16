import re
import time
from typing import Optional, cast

import requests
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from loguru import logger

from core.state import (RawDocument, SearchedDocument,
                        SearchQueriesStructureOutput, SearchWorkflowState)

from .base_agent import BaseAgent, BaseAgentOutput, BaseAgentState
from .prompts import (SITE_INFO_EXTRACTION_TEMPALTE, get_current_date,
                      query_writer_instructions)

options = ConversionOptions()
options.extract_metadata = False
options.autolinks = False


class ResearchAgent(BaseAgent):
    def __init__(self, model_name: str, max_result: int, n_queries: int):
        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
        self.model_search_queries = self.model.with_structured_output(
            SearchQueriesStructureOutput
        )
        self.n_queries = n_queries
        self.max_result = max_result
        self._compiled_graph = None

        logger.info(
            f"[{self.name}] 🔧 Agent initialize with {model_name}, max_result={max_result}"
        )

    @property
    def name(self):
        return "WEB_RESEARCHER - 'The Knowledge Retrieval System'"

    @property
    def purpose(self):
        purpose = """
        Gather, analyze, and summarize information from the internet
        Capabilities:
        - Web search and information retrieval
        - Multi-source data synthesis
        - Fact-checking and source verification
        - Summarization and contextual analysis
        Use when: Task requires current information, research, or data not in training set
        """
        return purpose

    async def create_search_queries(self, state: SearchWorkflowState):
        context = query_writer_instructions.format(
            number_queries=self.n_queries,
            current_date=get_current_date(),
            research_topic=state["workflow_input"],
        )
        response: SearchQueriesStructureOutput = await self.model_search_queries.ainvoke(context)  # type: ignore
        logger.info(
            f"[{self.name}] 🔍 The following search queries were selected: {response.query[:self.n_queries]}"
        )
        return {"search_queries": response.query[: self.n_queries]}

    def searching(self, state: SearchWorkflowState) -> SearchWorkflowState:
        results = []
        processed_count = 0
        failed_count = 0

        for query in state["search_queries"]:
            logger.info(
                f"[{self.name}] 🔍 Start searching: '{query}' (max results: {self.max_result})"
            )

            try:
                search_results = DDGS().text(query, max_results=self.max_result)  # type: ignore
                logger.info(
                    f"[{self.name}] 📊 Received  {len(search_results)} results from the search engine"
                )
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Error while searching: {e}")
                time.sleep(1)
                continue

            for idx, x in enumerate(search_results, 1):
                time.sleep(0.5)
                url: Optional[str] = x.get("href", None)
                title: Optional[str] = x.get("title", "Без заголовка")

                if url is None:
                    logger.warning(
                        f"[{self.name}] ⚠️ Resultт #{idx}: URL is missing, skip"
                    )
                    continue

                try:
                    r = requests.get(
                        url,
                        headers={
                            "User-Agent": "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
                        },
                        timeout=10,
                    )

                    if r.status_code == 200:
                        document: RawDocument = {"url": url, "source": r.text}
                        results.append(document)
                        processed_count += 1
                        logger.success(
                            f"[{self.name}] ✅ Successfully loaded: {url} ({len(r.text)} characters)"
                        )
                    else:
                        logger.warning(
                            f"[{self.name}] ⚠️ HTTP {r.status_code} for {url}"
                        )
                        failed_count += 1

                except requests.exceptions.Timeout:
                    logger.error(f"[{self.name}] ⏰ Tieout while loading {url}")
                    failed_count += 1
                except requests.exceptions.RequestException as e:
                    logger.error(f"[{self.name}] ❌ Network exception for {url}: {e}")
                    failed_count += 1
                except Exception as e:
                    logger.error(f"[{self.name}] ❌ exception for {url}: {e}")
                    failed_count += 1
            time.sleep(1)

        state["sources"] = results
        logger.info(
            f"[{self.name}] 📋 Search results: {processed_count} successful, {failed_count} with errors, total {len(results)} sources"
        )
        return state

    async def extract_text_from_search(self, state: SearchWorkflowState):
        contexts = []
        for doc in state["sources"]:
            url = doc["url"]
            logger.info(f"[{self.name}] 🧠 Starting to extract information from: {url}")

            try:
                markdown = convert(doc["source"], options)
                markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
                contexts.append(SITE_INFO_EXTRACTION_TEMPALTE.format(markdown=markdown))

            except Exception as e:
                logger.error(
                    f"[{self.name}] ❌ Error turning into markdown from {url}: {e}"
                )

        start_time = time.time()
        responses = await self.model.abatch(contexts)
        processing_time = time.time() - start_time

        results = []

        for response, source in zip(responses, state["sources"]):
            logger.info(
                f"[{self.name}] ✅ Extracted {len(response.content)} characters in {processing_time:.2f} seconds"
            )
            logger.debug(
                f"[{self.name}] 📝 Extracted result (first 500 characters): {response.content[:500]}..."
            )

            document: SearchedDocument = {
                "url": source["url"],
                "source": source["source"],
                "extracted_info": cast(str, response.content),
            }
            results.append(document)

        return {"searched_documents": results}

    def transform_to_output(self, state: SearchWorkflowState) -> BaseAgentOutput:
        return {
            "output": "\n\n".join(
                [x["extracted_info"] for x in state["searched_documents"]]
            )
        }

    def build_graph(self) -> StateGraph:
        try:
            builder = StateGraph(
                SearchWorkflowState,
                input_schema=BaseAgentState,
                output_schema=BaseAgentOutput,
            )

            builder.add_node("create_search_queries", self.create_search_queries)
            builder.add_node("searching", self.searching)
            builder.add_node("extract_info", self.extract_text_from_search)
            builder.add_node("transform_to_output", self.transform_to_output)

            builder.set_entry_point("create_search_queries")

            builder.add_edge("create_search_queries", "searching")

            builder.add_edge("searching", "extract_info")

            builder.add_edge("extract_info", "transform_to_output")
            builder.add_edge("transform_to_output", END)

            logger.success(
                f"[{self.name}] ✅ The workflow graph has been successfully built"
            )

            return builder

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error while constructing graph: {e}")
            raise
