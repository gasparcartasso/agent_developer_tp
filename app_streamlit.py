import hmac
import streamlit as st
from app.src.triage_graph import generate_graph
from langgraph.types import Command
import uuid

def check_login():
    if st.session_state.get("authenticated"):
        return True

    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        expected = st.secrets["passwords"].get(user)
        if expected and hmac.compare_digest(pwd, expected):
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Invalid username or password")
    return False

if not check_login():
    st.stop()  # nothing below runs until logged in

st.sidebar.write(f"Logged in as **{st.session_state['user']}**")
if st.sidebar.button("Log out"):
    st.session_state.clear()
    st.rerun()

st.title("Email triage and response")

@st.cache_resource
def get_agent():
    return generate_graph()

super_agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"{st.session_state['user']}-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you with your emails today?"}
    ]
if "pending" not in st.session_state:
    st.session_state["pending"] = None

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

def add_assistant(text):
    st.session_state.messages.append({"role": "assistant", "content": text})

# 1. History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 2. Review UI, rendered on every rerun while a draft is waiting
pending = st.session_state["pending"]
if pending:
    with st.chat_message("assistant"):
        st.markdown("**Draft reply: please review before it is sent**")
        st.caption(f"Member number: {pending.get('member_number')}")
        edited = st.text_area("Draft (you can edit it)", value=pending["draft_response"], height=260)
        col_approve, col_edit, col_decline = st.columns(3)
        decision = None
        if col_approve.button("Approve & send"):
            decision = {"human_decision": "approved"}
        if col_edit.button("Send edited version"):
            decision = {"human_decision": "edited", "draft_response": edited}
        if col_decline.button("Decline"):
            decision = {"human_decision": "declined"}

    if decision:
        try:
            with st.spinner("Sending..."):
                result = super_agent.invoke(Command(resume=decision), config=config)
            add_assistant(str(result.get("final_response", result)))
        except Exception as e:
            add_assistant(f"Sorry, I ran into an error sending that: {e}")
        st.session_state["pending"] = None
        st.rerun()

# 3. New prompt (disabled while a draft awaits review)
if prompt := st.chat_input(disabled=pending is not None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Thinking..."):
            response = super_agent.invoke(
                {"email_body": prompt, "user": st.session_state["user"],
                 "thread": st.session_state["thread_id"], "project": "kb"},
                config=config,
            )
        if "__interrupt__" in response:
            st.session_state["pending"] = response
        elif response.get("is_question_about_policy") is False:
            add_assistant("This email is not about policy or coverage. No further action is needed.")
        else:
            add_assistant(str(response.get("final_response", response)))
    except Exception as e:
        add_assistant(f"Sorry, I ran into an error answering that: {e}")
    st.rerun()