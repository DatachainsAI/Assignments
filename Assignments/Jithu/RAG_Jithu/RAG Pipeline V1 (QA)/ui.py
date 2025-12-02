import streamlit as st
from banking_rag_qa import ask

st.title("🔍 Banking Compliance RAG Assistant")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        answer = ask(question)
        st.write("### Answer:")
        st.write(answer)
