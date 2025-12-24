from dataclasses import dataclass


@dataclass
class ResearchAgentConfig:
    """Configuration for the coding agent."""

    max_result: int
    n_queries: int

    model_name: str

    name: str = "WEB_RESEARCHER"
    puprose: str = """Gather, analyze, and summarize information from the internet
Capabilities:
- Web search and information retrieval
- Multi-source data synthesis
- Fact-checking and source verification
- Summarization and contextual analysis
Use when: Task requires current information, research, or data not in training set"""

    additional_input_prompt: str = """- workflow_input: Frame as research questions: "Research [specific_topic] focusing on [aspects]. Find [type_of_information] and verify [key_facts]."
- context: Include ONLY messages containing:
  - Research topics or information needs
  - Questions requiring factual answers
  - Requests for current information or data not in training set
  - NEVER include pure computational tasks or code requirements"""

    user_agent: str = (
        "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
    )
