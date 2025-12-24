from typing import Optional

from .code_interfaces import ICodeAnalyzer, ICodeGenerator, ICodeReviewer
from .code_config import CodingAgentConfig
from .coding_agent import CodingAgent


class CodingAgentFactory():
    @staticmethod
    def create(
        config: CodingAgentConfig,
        analyzer: Optional[ICodeAnalyzer] = None,
        generator: Optional[ICodeGenerator] = None,
        reviewer: Optional[ICodeReviewer] = None
    ):
        return CodingAgent(
            config=config,
            analyzer=analyzer,
            generator=generator,
            reviewer=reviewer
        )

    @staticmethod
    def create_default():
        config = CodingAgentConfig(model_name="llama3.1:8b")
        return CodingAgent(config=config)


#DEFAULT AGENT
coding_agent = CodingAgentFactory.create_default()