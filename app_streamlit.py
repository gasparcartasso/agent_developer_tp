import streamlit as st

if not st.user.is_logged_in:
    st.title("Please log in")
    st.button("Log in with Google", on_click=st.login)
    # multiple providers: st.login("google") / st.login("microsoft")
    st.stop()

st.sidebar.write(f"Hello, **{st.user.name}**")
st.sidebar.button("Log out", on_click=st.logout)

st.title("Protected app")
st.write(st.user.email)