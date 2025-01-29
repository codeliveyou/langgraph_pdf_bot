import glob

import streamlit as st
from dotenv import load_dotenv

from src.graph import GraphWorkflow
import src.utils as utz
import src.nodes as nodes

load_dotenv()

@st.cache_resource
def initialize_embeddings_and_docs():
    embeddings = utz.initialize_embeddings()
    docs = utz.load_and_process_pdfs("./docs")
    doc_splits = utz.setup_document_processing(docs)
    ensemble_retriever = nodes.initialize_retriever(doc_splits, embeddings)
    return ensemble_retriever

def chatbot_page():

    st.title("Langgraph Chatbot")
    st.write("You can ask questions what you want to know")

    graph = GraphWorkflow()
    app = graph.app

    user_input = st.text_area("Ask a question...", height=100)

    if user_input:
        with st.spinner("Answering your question..."):
            # Running the graph with the user's question
            inputs = {"question": user_input}
            result = None

            try:
                for output in app.stream(inputs):
                    for key, value in output.items():
                        if "generation" in value:
                            result = value["generation"]

            except Exception as e:
                st.error(f"Error occurred: {e}")

        if result:
            st.write(f"\nAnswer: {result}\n")
        else:
            print("\nSorry, I couldn't find an answer to that question.\n")


if __name__ == "__main__":

    ensemble_retriever = initialize_embeddings_and_docs()

    chatbot_page()