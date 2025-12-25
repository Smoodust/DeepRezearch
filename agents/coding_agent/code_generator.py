from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from loguru import logger

from core.template_manager import TemplateManager

from .code_state import Code, CodeAnalysis
from .code_interfaces import ICodeGenerator


class CodeGenerator(ICodeGenerator):
    def __init__(
        self,
        chat: type[BaseChatModel],
        model_name: str,
        temperature: float,
        num_predict: int,
        tools: list[Tool],
        checkpointer: Optional[BaseCheckpointSaver[str]] = None,
    ):
        if checkpointer is None or isinstance(checkpointer, type):
            checkpointer = InMemorySaver()

        self.checkpointer = checkpointer
        self.tools = tools
        self.name = "CodeGenerator"

        model = chat(
            model=model_name,
            format="json",
            temperature=temperature,
            num_predict=num_predict,
        )

        self.generation_agent = create_agent(
            model=model,
            tools=self.tools,
            system_prompt=TemplateManager().render_template(
                "coding_agent/CODE_GENERATION_PROMPT.jinja"
            ),
            checkpointer=self.checkpointer,
        )

        self.thread_id: Optional[str] = None

    def set_thread_id(self, thread_id: str) -> None:
        self.thread_id = thread_id

    async def generate(
        self, task: str, analysis: CodeAnalysis, feedback: Optional[list[str]] = None
    ) -> Code:
        if not analysis:
            raise ValueError("Analysis cannot be None")

        generation_prompt = TemplateManager().render_template(
            "coding_agent/CODE_GENERATION_TEMPLATE.jinja",
            task=task,
            plan=analysis.model_dump(),
            requirements=analysis.requirements,
            feedback=feedback[-3:] if feedback else [],
        )

        logger.debug(f"[{self.name}] 🔄 Start generating code")

        messages = []

        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            agent_input = {"messages": [HumanMessage(content=generation_prompt)]}
            response: dict = await self.generation_agent.ainvoke(
                agent_input, config=config
            )

            messages = response.get("messages", [])

            tool_output = next(
                (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
                "[No output produced]",
            )

            ai_messsage = next(
                (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
                "[No output produced]",
            )

            code_str = ai_messsage

            logger.debug(f"{code_str}\n{tool_output}")

            logger.success(f"[{self.name}] ✅ Generated and executed code")

            return Code(code=code_str, output=tool_output)

        except GraphRecursionError:
            logger.error(
                f"[{self.name}] ❌ Agent stuck in infinite loop. Check agent configuration or task complexity."
            )

            state = self.generation_agent.get_state(config)
            result = state.values
            messages = result.get("messages", [])

            print(messages)

            tool_output = next(
                (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
                "[No output produced]",
            )

            ai_messsage = next(
                (m.tool_calls for m in reversed(messages) if isinstance(m, AIMessage)),
                "[No output produced]",
            )

            code_str = ""

            for tc in ai_messsage:
                if tc.get("name") == "python_repl":
                    args = tc.get("args", {})
                    if "__arg1" in args:
                        code_str = args["__arg1"]

            logger.debug(f"{code_str}\n{tool_output}")

            return Code(
                code=code_str,
                output=tool_output,
            )

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Critical generation failure: {str(e)}")
            logger.debug(f"[{self.name}] Input prompt:\n{generation_prompt}")

            if "code_str" not in locals():
                code_str = "[Code generation failed]"

            return Code(code=code_str, output=str(e))

    async def delete_thread_id(self) -> None:
        if self.thread_id:
            try:
                await self.checkpointer.adelete_thread(self.thread_id)
            except Exception as e:
                logger.warning(
                    f"[{self.name}] Failed to cleanup thread {self.thread_id}: {e}"
                )
            finally:
                self.thread_id = None
