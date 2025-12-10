from base_agent import BaseAgent


class SynthesisAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "synthesis"

    @property
    def purpose(self) -> str:
        return "Makes final answer"
