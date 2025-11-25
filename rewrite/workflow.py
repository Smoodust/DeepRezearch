import operator

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from prompt import *
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict, cast

model_name = "deepseek-r1:7b"
llm = init_chat_model(model_name, model_provider="ollama")


class OrchestratorAnswer(BaseModel):
    reasoning: str = Field(
        description="Brief thinking and explanation of classification. You should reason why you choose this workflow.",
    )
    workflow_type: Literal["coding", "search", "synthesize"] = Field(
        None, description="The next chosen workflow in the process" #type:ignore
    )
    workflow_task: str = Field(
        description="Explanation of what this workflow would do. If search then what search query should they use and researh overall. If coding what code they need to write."
    )


orchestrator = llm.with_structured_output(OrchestratorAnswer)


class State(TypedDict):
    request: str
    garbage_context: Annotated[list[str], operator.add]
    next_step: Literal["coding", "search", "synthesize"]
    next_workflow_task: str
    final_report: str


def make_context(system_prompt, user_prompt, garbage_context, n=3):
    context = [SystemMessage(system_prompt)]
    context += [HumanMessage(x) for x in garbage_context[-n:]]
    context.append(HumanMessage(user_prompt))
    return context


def llm_orchestrator(state: State):
    context = make_context(
        orchestrator_system_prompt,
        orchestrator_user_prompt.format(user_input=state["request"]),
        state["garbage_context"][-3:],
    )
    decision = cast(OrchestratorAnswer, orchestrator.invoke(context))
    print(decision)
    return {"next_step": decision.workflow_type, "next_workflow_task": decision.workflow_task}


def coding(state: State):
    return {}


def searching(state: State):
    return {}


def synthesizer(state: State):
    context = make_context(
        synthesizer_system_prompt,
        synthesizer_user_prompt.format(user_input=state["request"]),
        state["garbage_context"][-3:],
    )
    return {"final_report": llm.invoke(context).content}


def should_continue(state: State):
    return state["next_step"]


deep_researcher_builder = StateGraph(State)
deep_researcher_builder.add_node("orchestrator", llm_orchestrator)
deep_researcher_builder.add_node("coding", coding)
deep_researcher_builder.add_node("searching", searching)
deep_researcher_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
deep_researcher_builder.add_edge(START, "orchestrator")
deep_researcher_builder.add_conditional_edges(
    "orchestrator",
    should_continue,
    {"coding": "coding", "search": "searching", "synthesizer": "synthesizer"},
)
deep_researcher_builder.add_edge("coding", "orchestrator")
deep_researcher_builder.add_edge("searching", "orchestrator")
deep_researcher_builder.add_edge("synthesizer", END)

# Compile the workflow
deep_researcher = deep_researcher_builder.compile()
