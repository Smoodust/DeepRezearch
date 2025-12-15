import re
import time
from typing import Optional, cast

import requests
from ddgs import DDGS
from html_to_markdown import ConversionOptions, convert
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.types import Send
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
            f"[{self.name}] 🔧 Агент инициализирован с моделью {model_name}, max_result={max_result}"
        )

    @property
    def name(self):
        return "research"

    @property
    def purpose(self):
        return "For information gathering: research, analysis, data collection, comparative studies."

    async def create_search_queries(self, state: SearchWorkflowState):
        context = query_writer_instructions.format(
            number_queries=self.n_queries,
            current_date=get_current_date(),
            research_topic=state["workflow_input"],
        )
        response: SearchQueriesStructureOutput = await self.model.ainvoke(context)  # type: ignore
        logger.info(
            f"[{self.name}] 🔍 Были выбраны следующие запросы для поиска: {response.query[:self.n_queries]}"
        )
        return {"search_queries": response.query[: self.n_queries]}

    def searching(self, state: SearchWorkflowState) -> SearchWorkflowState:
        results = []
        processed_count = 0
        failed_count = 0

        for query in state["search_queries"]:
            logger.info(
                f"[{self.name}] 🔍 Начинаю поиск по запросу: '{query}' (макс. результатов: {self.max_result})"
            )

            try:
                search_results = DDGS().text(query, max_results=self.max_result)  # type: ignore
                logger.info(
                    f"[{self.name}] 📊 Получено {len(search_results)} результатов от поисковика"
                )
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Ошибка при поиске: {e}")
                time.sleep(1)
                continue

            for idx, x in enumerate(search_results, 1):
                time.sleep(0.5)
                url: Optional[str] = x.get("href", None)
                title: Optional[str] = x.get("title", "Без заголовка")

                if url is None:
                    logger.warning(
                        f"[{self.name}] ⚠️ Результат #{idx}: URL отсутствует, пропускаю"
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
                            f"[{self.name}] ✅ Успешно загружено: {url} ({len(r.text)} символов)"
                        )
                    else:
                        logger.warning(
                            f"[{self.name}] ⚠️ HTTP {r.status_code} для {url}"
                        )
                        failed_count += 1

                except requests.exceptions.Timeout:
                    logger.error(f"[{self.name}] ⏰ Таймаут при загрузке {url}")
                    failed_count += 1
                except requests.exceptions.RequestException as e:
                    logger.error(f"[{self.name}] ❌ Ошибка сети для {url}: {e}")
                    failed_count += 1
                except Exception as e:
                    logger.error(f"[{self.name}] ❌ Неожиданная ошибка для {url}: {e}")
                    failed_count += 1
            time.sleep(1)

        state["sources"] = results
        logger.info(
            f"[{self.name}] 📋 Итог поиска: {processed_count} успешно, {failed_count} с ошибками, всего {len(results)} источников"
        )
        return state

    async def extract_text_from_search(self, state: RawDocument):
        url = state["url"]
        logger.info(f"[{self.name}] 🧠 Начинаю извлечение информации из: {url}")

        try:
            markdown = convert(state["source"], options)

            markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)

            context = SITE_INFO_EXTRACTION_TEMPALTE.format(markdown=markdown)

            start_time = time.time()
            response = await self.model.ainvoke(context)
            processing_time = time.time() - start_time

            extracted_length = len(cast(str, response.content))
            logger.info(
                f"[{self.name}] ✅ Извлечено {extracted_length} символов за {processing_time:.2f} сек"
            )
            logger.debug(
                f"[{self.name}] 📝 Результат извлечения (первые 500 символов): {response.content[:500]}..."
            )

            document: SearchedDocument = {
                "url": url,
                "source": state["source"],
                "extracted_info": cast(str, response.content),
            }
            return {"searched_documents": [document]}

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка при извлечении из {url}: {e}")
            document: SearchedDocument = {
                "url": url,
                "source": state["source"],
                "extracted_info": f"Ошибка при извлечении: {str(e)}",
            }
            return {"searched_documents": [document]}

    def search_map(self, state: SearchWorkflowState):
        len(state["sources"])

        sends = [Send("extract_info", d) for d in state["sources"]]
        logger.debug(f"[{self.name}] 📤 Создано {len(sends)} задач для обработки")

        for i, d in enumerate(state["sources"][:3]):
            logger.debug(f"[{self.name}] 📎 Источник #{i+1}: {d['url'][:100]}...")

        return sends

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

            builder.add_conditional_edges(
                "searching", self.search_map, ["extract_info"]
            )

            builder.add_edge("extract_info", "transform_to_output")
            builder.add_edge("transform_to_output", END)

            logger.success(f"[{self.name}] ✅ Workflow граф успешно построен")

            return builder

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка при построении графа: {e}")
            raise
