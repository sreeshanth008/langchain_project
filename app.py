import streamlit as st 
from components import MasterAgent

st.title("Langchain Components")
a=st.chat_input("Ask a question:") 
sidebar=st.sidebar.title("Chat History")
master=MasterAgent("bDSwa5u31no621tcFnhyQvMWLnUDGVKQ")

    
    
if "history" not in st.session_state:
    st.session_state.history = []
for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)    


for role, content in st.session_state.history:
    if role=="user":
        st.sidebar.container(border=True).button(f"*User:* {content}")
 
if a:
    st.chat_message("user").markdown(a)
    st.sidebar.container(border=True).button(f"*User:* {a}")
    
    st.session_state.history.append(("user", a))
    with st.spinner("Thinking..."):
        response = master.perform_task(a)
    st.chat_message("assistant").markdown(response)
    st.session_state.history.append(("assistant", response))