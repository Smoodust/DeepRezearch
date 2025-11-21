from abc import ABC, abstractmethod

from langgraph.graph import StateGraph


class BaseWorkflow(ABC):
    @abstractmethod
    def _build_workflow(self):
        return

    @abstractmethod
    def reflection():
        pass

    @abstractmethod
    def evaluate():
        pass

    @abstractmethod
    def finalize_answer():
        pass


class ResearchWorkflow(BaseWorkflow):
    def __init__(self, model):
        self.model = model
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph()

        # some nodes

        return workflow.compile()

    def generate_query():
        pass

    def web_research():
        pass

    def continue_to_web_research():
        pass

    def reflection():
        pass

    def evaluate():
        pass

    def finalize_answer():
        pass


class CodeWorkflow(BaseWorkflow):
    def __init__(self, model):
        self.model = model
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph()

        # some nodes

        return workflow.compile()

    def generate_code():
        pass

    def reflection():
        pass

    def evaluate():
        pass

    def finalize_answer():
        pass


class SynthesisWorkflow(BaseWorkflow):
    def __init__(self, model):
        self.model = model
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph()

        # some nodes

        return workflow.compile()

    def synthesis_answer():
        pass

    def reflection():
        pass

    def evaluate():
        pass

    def finalize_answer():
        pass
