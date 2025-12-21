import asyncio
import json
import re
import time
from datetime import datetime
from typing import Optional, cast

import aiohttp
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from loguru import logger

from core.state import (RawDocument, SearchedDocument,
                        SearchQueriesStructureOutput, SearchWorkflowState)

from .base_agent import BaseAgent, BaseAgentOutput, BaseAgentState

options = ConversionOptions()
options.extract_metadata = False
options.autolinks = False


class ResearchAgent(BaseAgent):
    def __init__(self, model_name: str, max_result: int, n_queries: int):

        super().__init__()

        self.model_name = model_name
        self.model = init_chat_model(model_name, model_provider="ollama")
        self.model_search_queries = self.model.with_structured_output(
            SearchQueriesStructureOutput
        )

        self.n_queries = n_queries
        self.max_result = max_result

        self.user_agent = {
            "User-Agent": "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
        }

        self._query_writer_tpl = None
        self._site_info_tpl = None
        self._final_summary_tpl = None
        self._final_answer_tpl = None

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
    
    @property
    def additional_input_prompt(self) -> str:
        return """workflow_input: MUST be a focused research query. Include:
- Specific questions to answer: "Find...", "Research..."
- Required information types: "Statistics...", "Current events...", "Scientific studies..."
- Scope constraints: "Focus on...", "Exclude..."
- Output requirements: "Summarize...", "Compare..."

context: Include:
- User's original question
- Any known facts to verify/expand upon
- Timeframe requirements: "Current as of...", "Historical..."
- Credibility requirements: "Peer-reviewed sources...", "Official data...\""""
    
    @property
    def examples_input_prompt(self) -> str:
        return """For WEB_RESEARCHER (User: "What's the latest SpaceX launch?"):
```json
{
    "thinking": "WEB_RESEARCHER needs a clear search query focused on recent SpaceX activities",
    "workflow_input": "Research the most recent SpaceX rocket launch. Find details including: mission name, launch date, payload, launch site, and mission outcome. Also check for upcoming scheduled launches.",
    "context": "User wants current information about SpaceX launches. Focus on reliable space news sources and official SpaceX communications. Information should be from the past 30 days."
}
```"""

    # Get current date in a readable format
    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%B %d, %Y")

    async def create_search_queries(self, state: SearchWorkflowState):
        if self._query_writer_tpl is None:
            self._query_writer_tpl = self._load_template(
                "research_agent/QUERY_WRITER.jinja"
            )

        context = self._query_writer_tpl.render(
            number_queries=self.n_queries,
            current_date=self.get_current_date(),
            research_topic=state["workflow_input"],
        )

        try:
            response: SearchQueriesStructureOutput = await self.model_search_queries.ainvoke(context)  # type: ignore
            logger.info(
                f"[{self.name}] 🔍 The following search queries were selected: {response.query[:self.n_queries]}"
            )
            return {"search_queries": response.query[: self.n_queries]}
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error in creating search queries: {e}")
            return {"search_queries": []}

    async def searching(self, state: SearchWorkflowState) -> SearchWorkflowState:
        results = []

        connector = aiohttp.TCPConnector(limit_per_host=5, force_close=True)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        processed_count = 0
        failed_count = 0

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self.user_agent
        ) as session:
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
                    await asyncio.sleep(1)
                    continue

                for idx, x in enumerate(search_results, 1):
                    await asyncio.sleep(0.5)
                    url: Optional[str] = x.get("href", None)
                    title: Optional[str] = x.get("title", "Без заголовка")

                    if url is None:
                        logger.warning(
                            f"[{self.name}] ⚠️ Resultт #{idx}: URL is missing, skip"
                        )
                        continue

                    try:
                        r = await session.get(url)

                        if r.status == 200:
                            html_content = await r.text()

                            document: RawDocument = {"url": url, "source": html_content}
                            results.append(document)
                            processed_count += 1
                            logger.success(
                                f"[{self.name}] ✅ Successfully loaded: {url} ({len(html_content)} characters)"
                            )
                        else:
                            logger.warning(f"[{self.name}] ⚠️ HTTP {r.status} for {url}")
                            failed_count += 1

                    except aiohttp.ServerTimeoutError:
                        logger.error(f"[{self.name}] ⏰ Tieout while loading {url}")
                        failed_count += 1
                    except aiohttp.ClientConnectorError as e:
                        logger.error(
                            f"[{self.name}] ❌ Network exception for {url}: {e}"
                        )
                        failed_count += 1
                    except Exception as e:
                        logger.error(f"[{self.name}] ❌ exception for {url}: {e}")
                        failed_count += 1
                await asyncio.sleep(1)

        state["sources"] = results
        logger.info(
            f"[{self.name}] 📋 Search results: {processed_count} successful, {failed_count} with errors, total {len(results)} sources"
        )
        return state

    async def extract_text_from_search(self, state: SearchWorkflowState):
        if self._site_info_tpl is None:
            self._site_info_tpl = self._load_template(
                "research_agent/SITE_INFO_INSTRUCTIONS.jinja"
            )

        contexts = []

        for doc in state["sources"]:
            url = doc["url"]
            logger.info(f"[{self.name}] 🧠 Starting to extract information from: {url}")

            try:
                markdown = convert(doc["source"], options)
                markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
                contexts.append(
                    self._site_info_tpl.render(markdown=markdown, workflow_input=state["workflow_input"])
                )

            except Exception as e:
                logger.error(
                    f"[{self.name}] ❌ Error turning into markdown from {url}: {e}"
                )

        start_time = time.time()

        try:
            responses: list = await self.model.abatch(contexts)
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

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error during information extraction: {e}")
            return {"searched_documents": []}

    async def summary_to_output(self, state: SearchWorkflowState) -> BaseAgentOutput:
        if self._final_summary_tpl is None:
            self._final_summary_tpl = self._load_template(
                "research_agent/FINAL_SUMMARY_PROMPT.jinja"
            )
        
        if self._final_answer_tpl is None:
            self._final_answer_tpl = self._load_template(
                "research_agent/FINAL_ANSWER_TEMPLATE.jinja"
            )

        documents = [{"id": id+1, "url": x["url"], "text": x["extracted_info"]} for id, x in enumerate(state["searched_documents"])]
        prompt_docs = [{"id": x["id"], "text": x["text"]} for x in documents]
        prompt = self._final_summary_tpl.render(documents=json.dumps(prompt_docs, indent=4), workflow_input=state["workflow_input"])
        summary: str = (await self.model.ainvoke(prompt)).content #type: ignore

        return {
            "output": self._final_answer_tpl.render(summary=summary, documents=documents)
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
            builder.add_node("summary_to_output", self.summary_to_output)

            builder.set_entry_point("create_search_queries")

            builder.add_edge("create_search_queries", "searching")

            builder.add_edge("searching", "extract_info")

            builder.add_edge("extract_info", "summary_to_output")
            builder.add_edge("summary_to_output", END)

            logger.success(
                f"[{self.name}] ✅ The workflow graph has been successfully built"
            )

            return builder

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error while constructing graph: {e}")
            raise
