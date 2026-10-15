import hmac
import streamlit as st
from app.src.agent_builder import SuperAgent

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
super_agent = SuperAgent()
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you with your emails today?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = super_agent.ask(prompt,user=st.session_state["user"], thread="thread1", project="kb")
                answer = response["output"]
            except Exception as e:
                answer = f"Sorry, I ran into an error answering that: {e}"
        try:
          st.write(answer[0]['text'])
          st.session_state.messages.append({"role": "assistant", "content": answer[0]['text']})
        except:
          st.write(answer)