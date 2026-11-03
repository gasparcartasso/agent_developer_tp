from typing import TypedDict,Literal
 
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from app.src.agent_search_knowleadge import SearchKnowledgeBaseAgent
from app.src.agent_member_info import SearchMemberInfoAgent

llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
    num_ctx=16384,                      # Ollama's default window is small
    reasoning=False
    #client_kwargs={"timeout": 60},
)

class EmailState(TypedDict, total=False):
    email_body: str
    user: str | None
    thread: str | None
    project: str | None
    member_number: int | None
    is_question_about_policy: bool
    knowledge_base_results: str
    member_info: str
    draft_response: str
    human_decision: str
    email_sent: bool

class Triage(BaseModel):
    member_number: int = Field(description="The member number extracted from the email")
    is_question_about_policy: bool = Field(description="Whether the email is asking about policy or coverage")

llm_triage = llm.with_structured_output(Triage)

def node_triage_email(state:EmailState) -> EmailState:
    triage_result = llm_triage.invoke(state["email_body"])
    return {'member_number': triage_result.member_number, 'is_question_about_policy': triage_result.is_question_about_policy}

def node_search_knowledge_base(state:EmailState) -> EmailState:
    search_policy = SearchKnowledgeBaseAgent()
    knowledge_base_results = search_policy.ask(f"{state["email_body"]}\n\n{state['member_info']}", user=state.get("user"), thread=state.get("thread"), project=state.get("project"))
    return {'knowledge_base_results': knowledge_base_results.get("final_answer")}

def node_search_member_info(state:EmailState) -> EmailState:
    search_member_info = SearchMemberInfoAgent()
    member_info = search_member_info.ask(f"Search this member number {state["member_number"]} and retrieve their information", user=state.get("user"), thread=state.get("thread"), project=state.get("project"))
    return {'member_info': member_info.get("final_answer")}

def node_draft_response(state:EmailState) -> EmailState:
    llm_draft = llm.invoke(f"Draft a response to this email: {state['email_body']}\n\nUse the following information to help you answer: {state['knowledge_base_results']}\n\n{state['member_info']}")
    return {'draft_response': llm_draft}

def human_in_the_loop(state:EmailState) -> EmailState:
    decision = interrupt({
        "instruction": "Approve, decline or edit this content",
        "content": state["draft_response"],
    })
    if decision.get("human_decision") == "edited":
        return {"human_decision": decision.get("human_decision"), "draft_response": decision.get("draft_response")}
    if decision.get("human_decision") == "approved":
        return {"human_decision": decision.get("human_decision")}
    if decision.get("human_decision") == "declined":
        return {"human_decision": decision.get("human_decision")}

def route_question_triage(state: EmailState) -> str:
    if state["is_question_about_policy"]:
        return "search_member_info"
    return END

def node_send_email(state:EmailState) -> EmailState:
    # Here you would implement the logic to send the email using your preferred method (e.g., SMTP, API call, etc.)
    # For demonstration purposes, we'll just return a success message.
    return {"email_sent": True}

def route_hitl(state: EmailState) -> str:
    if state.get("human_decision") == "approved":
        return "send_email"
    elif state.get("human_decision") == "edited":
        return "send_email"
    elif state.get("human_decision") == "declined":
        return END

def generate_graph():
    graph = StateGraph(EmailState)
    graph.add_node("triage_email", node_triage_email).add_node("search_knowledge_base", node_search_knowledge_base).add_node("search_member_info", node_search_member_info).add_node("draft_response", node_draft_response).add_node("human_in_the_loop", human_in_the_loop).add_node("send_email", node_send_email)
    graph.add_edge(START,"triage_email")
    graph.add_conditional_edges("triage_email", route_question_triage)
    graph.add_edge('search_member_info',"search_knowledge_base")
    graph.add_edge('search_knowledge_base',"draft_response")
    graph.add_edge("draft_response", "human_in_the_loop")
    graph.add_conditional_edges("human_in_the_loop", route_hitl)
    graph.add_edge("send_email", END)
    return graph.compile(InMemorySaver())