import hmac
import streamlit as st

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

st.title("Protected app")