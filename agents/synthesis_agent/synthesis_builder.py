from dataclasses import dataclass

from ..base_agent import BaseAgent, BaseAgentBuilder
from .synthesis_agent import SynthesisAgent


@dataclass
class SynthesisAgentBuilder(BaseAgentBuilder):
    """Configuration for the coding agent."""

    model_name: str

    name: str = "RESPONSE_SYNTHESIZER"
    purpose: str = """Synthesizes multi-source information into coherent, well-structured final responses
Capabilities:
- Integrates information from multiple agent outputs
- Formats responses for different audience types
- Structures complex information logically
- Maintains conversational tone and clarity
- Adheres to specified format requirements
 Use when:
- Preparing final responses from multi-agent workflows
- Combining research findings with computational result
- Handling conversational greetings and informal queries"""

    additional_input_prompt: str = """- workflow_input: Specify synthesis requirements: "Synthesize the [information] from provided [context] into [response_format] addressing [user's_original_query]. Focus on [key_points] and structure for [audience_type]."
- context: Include messages containing:
  - User's original question/request (usually first message)
  - Agent outputs with results/data (identified by their IDs)
  - Any formatting or tone requirements
  - NEVER include intermediate thinking or routing decisions"""

    def build(self) -> BaseAgent:
        return SynthesisAgent(
            self.name, self.purpose, self.additional_input_prompt, self.model_name
        )
