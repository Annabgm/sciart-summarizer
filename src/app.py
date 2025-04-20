import streamlit as st
import tempfile

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks import get_openai_callback

from llm_chains.rag import RAG
from llm_chains.spendings import SpendingClient, Spendings, SpendingsMeta, spend_helper
from llm_chains.citation_styles import SummaryCitation

st.set_page_config(page_title="Scientific Summarizer", layout="wide")

vectorstore = Chroma(collection_name="langchain", embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db")
spending_client = SpendingClient(client_name="spendings")
model = ChatOpenAI(temperature = 0.0, model="gpt-4o-mini")
sum_assistant = RAG(model, vectorstore, spending_client)
chain = sum_assistant.create_graph()

tab1, tab2 = st.tabs(["Summary", "Spendings"])

with tab1:
    st.header("Summary")
    # File uploader for PDF
    uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
    st.markdown("Slider below inderectly define size of summary. The more the value, the more the number of " +
                "chunks is used as context.")
    size = st.slider("Choose the size of context", 0, 100, 5)
    # Input for the question
    question = st.text_input("Enter your question about the document:")

    # Button to process the input
    if st.button("Generate Summary"):
        if uploaded_files and question:
            with tempfile.TemporaryDirectory() as temp_dir:  # Create a temporary directory
                for uploaded_file in uploaded_files:
                    # Save each uploaded file to the temporary directory
                    temp_file_path = f"{temp_dir}/{uploaded_file.name}"
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    # load the PDF and store it in the vectorstore
                    docs = sum_assistant.load_pdf(temp_file_path)
                    meta = sum_assistant.metadata_from_pdf(docs)
                    sum_assistant.store_pdf(docs, meta)
            with get_openai_callback() as cb:
                config = {"configurable": {"chunk_nums": size}}
                result = chain.invoke({"question": question}, config=config)
                spending_client.add_spending(Spendings(cost=SpendingsMeta.from_api_response(cb)))

            citation_client = SummaryCitation.parse_summary(result)
            summary = citation_client.style()
            # Display the result
            st.subheader("Summary")
            st.write(summary)
        else:
            st.error("Please upload a PDF file and enter a question.")

with tab2:
    data = spend_helper(spending_client)

    st.header("Spendings")
    st.write("Here are your recent spendings.")
    # Display spending information
    st.subheader("Spending Information")
    st.dataframe(data, use_container_width=True)
