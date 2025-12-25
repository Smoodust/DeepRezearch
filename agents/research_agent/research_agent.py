import asyncio
import json
import re
import time
from datetime import datetime
from typing import Optional, Type, cast

import aiohttp
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel

from core.config import MODEL_URL
from core.template_manager import TemplateManager

from ..base_agent import BaseAgent, BaseAgentOutput, BaseAgentState
from .research_state import (RawDocument, SearchedDocument,
                             SearchQueriesStructureOutput, SearchWorkflowState,
                             WebResearchPlan)

options = ConversionOptions()
options.extract_metadata = False
options.autolinks = False


class ResearchAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        name: str,
        purpose: str,
        additional_input_prompt: str,
        user_agent: str,
        n_queries: int,
        max_result: int,
    ):

        super().__init__()

        self.model_name = model_name
        self.model = init_chat_model(
            model_name, model_provider="ollama", base_url=MODEL_URL
        )
        self.model_search_queries = self.model.with_structured_output(
            SearchQueriesStructureOutput
        )

        self.user_agent = {"User-Agent": user_agent}

        self._name = name
        self._purpose = purpose
        self._additional_input_prompt = additional_input_prompt
        self.n_queries = n_queries
        self.max_result = max_result

        logger.info(
            f"[{self.name}] 🔧 Agent initialize with {self.model_name}, max_result={self.max_result}"
        )

    @property
    def name(self):
        return self._name

    @property
    def purpose(self):
        return self._purpose

    @property
    def additional_input_prompt(self) -> str:
        return self._additional_input_prompt

    @property
    def get_input_model(self) -> Type[BaseModel]:
        return WebResearchPlan

    # Get current date in a readable format
    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%B %d, %Y")

    async def create_search_queries(self, state: SearchWorkflowState):
        context = TemplateManager().render_template(
            "research_agent/QUERY_WRITER.jinja",
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
        contexts = []

        for doc in state["sources"]:
            url = doc["url"]
            logger.info(f"[{self.name}] 🧠 Starting to extract information from: {url}")

            try:
                markdown = convert(doc["source"], options)
                markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
                contexts.append(
                    TemplateManager().render_template(
                        "research_agent/SITE_INFO_INSTRUCTIONS.jinja",
                        markdown=markdown,
                        workflow_input=state["workflow_input"],
                    )
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
        documents = [
            {"id": id + 1, "url": x["url"], "text": x["extracted_info"]}
            for id, x in enumerate(state["searched_documents"])
        ]
        prompt_docs = [{"id": x["id"], "text": x["text"]} for x in documents]
        prompt = TemplateManager().render_template(
            "research_agent/FINAL_SUMMARY_PROMPT.jinja",
            documents=json.dumps(prompt_docs, indent=4),
            workflow_input=state["workflow_input"],
        )

        logger.info(prompt)

        summary: str = (await self.model.ainvoke(prompt)).content  # type: ignore
        logger.success(f"RESEARCHER OUTPUT:\n{summary}")

        output: BaseAgentOutput = {
            "output": TemplateManager().render_template(
                "research_agent/FINAL_ANSWER_TEMPLATE.jinja",
                summary=summary,
                documents=documents,
            )
        }

        logger.success(output)

        return output

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
