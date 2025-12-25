from dataclasses import dataclass

from orchestrator import WorkflowOrchestrator

from ..base_agent import BaseAgent, BaseAgentBuilder


@dataclass
class OrchestratorBuilder(BaseAgentBuilder):
    """Configuration for the coding agent."""

    model_name: str
    agents: list[BaseAgent]
    agents_to_build: list[BaseAgentBuilder]

    name: str = "ORCHESTRATOR"
    purpose: str = ""

    user_agent: str = (
        "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
    )

    def build(self) -> BaseAgent:
        created_agents = self.agents
        for x in self.agents_to_build:
            created_agents.append(x.build())
        return WorkflowOrchestrator(
            self.model_name, self.name, self.purpose, created_agents
        )
