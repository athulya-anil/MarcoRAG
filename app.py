#Website
 
import streamlit as st
from vertex_rag import run_rag  #wraps my RAG pipeline into a function

st.set_page_config(page_title="Vertex RAG Engine", page_icon="🔍")

st.title("Vertex RAG Engine 🔍")
st.write("Ask me anything about Google Vertex AI docs")

query = st.text_area("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Searching..."):
            answer = run_rag(query)
        st.subheader("💡 Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

