from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

model_name = "deepseek-r1:7b"
llm = init_chat_model(model_name, model_provider="ollama")


class State(TypedDict):
    pass


def orchestrator(state: State):
    return {}


def coding(state: State):
    return {}


def searching(state: State):
    return {}


def synthesizer(state: State):
    return {}


def should_continue(state: State):
    return "synthesizer"


deep_researcher_builder = StateGraph(State)
deep_researcher_builder.add_node("orchestrator", orchestrator)
deep_researcher_builder.add_node("coding", coding)
deep_researcher_builder.add_node("searching", searching)
deep_researcher_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
deep_researcher_builder.add_edge(START, "orchestrator")
deep_researcher_builder.add_conditional_edges(
    "orchestrator",
    should_continue,
    {"coding": "coding", "searching": "searching", "synthesizer": "synthesizer"},
)
deep_researcher_builder.add_edge("coding", "orchestrator")
deep_researcher_builder.add_edge("searching", "orchestrator")
deep_researcher_builder.add_edge("synthesizer", END)

# Compile the workflow
deep_researcher = deep_researcher_builder.compile()
