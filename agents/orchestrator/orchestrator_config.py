from dataclasses import dataclass


@dataclass
class OrchestratorAgentConfig:
    """Configuration for the coding agent."""

    model_name: str

    name: str = "ORCHESTRATOR"
    puprose: str = ""

    additional_input_prompt: str = ""

    user_agent: str = (
        "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
    )
