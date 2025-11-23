from abstract_wf import BaseWorkflow
from langgraph.graph import StateGraph


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


class WorkflowOrchestrator:
    def __init__(self):
        self.workflows: Dict[WorkflowType, BaseWorkflow] = {}
        self._router = self._create_router()