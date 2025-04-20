from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables.config import RunnableConfig

import os
from pathlib import Path

from .objects import Bibcitation, QuotedAnswer, State
from .prompt_templates import system_prompt_meta, system_prompt_rag
from .pdf_processing import preprocess_pdf, make_hash_from_metadata
from .spendings import Spendings, SpendingsMeta, SpendingClient
from .context_postprocessing import format_docs_with_id


class RAG:
    def __init__(self, model: ChatOpenAI, vectorstore: VectorStore, spendings_client: SpendingClient):
        """
        Initializes the RAG class with a PDF file and a question.

        """
        self.llm = model
        self.vectorstore = vectorstore
        self.spendings = spendings_client

    def load_pdf(self, path: str) -> list[Document]:
        """
        Loads the PDF file and splits it into chunks.

        Returns:
            list[Document]: List of Document objects representing the chunks of the PDF.
        """
        # pdf_path = Path.cwd() / "tmp" / path # 248_ftp.pdf tmp/nl501863u.pdf tmp/Vol_241_Sample_pages.pdf
        loader = PyPDFLoader(path) 
        docs = loader.load()
        return docs
    
    def metadata_from_pdf(self, docs: list[Document]) -> list[Document]:
        """
        Extracts metadata from the PDF file.

        Args:
            docs (list[Document]): List of Document objects representing the chunks of the PDF.

        Returns:
            list[Document]: List of Document objects with metadata.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_meta),
                ("human", "{question}"),
            ]
        )
        meta_cite_llm = self.llm.with_structured_output(Bibcitation)
        chain = prompt | meta_cite_llm
        with get_openai_callback() as cb:
            response = chain.invoke({"question": "Extract the bibliographic information from the scientific paper.", "text": docs[0].page_content})
            self.spendings.add_spending(Spendings(cost=SpendingsMeta.from_api_response(cb)))

        return response.__dict__
    
    def store_pdf(self, docs: list[Document], paper_meta: dict[str, str]) -> list[Document]:
        """
        Splits the PDF file into smaller chunks and stores them in the vectorstore.

        Args:
            docs (list[Document]): List of Document objects representing the chunks of the PDF.

        Returns:
            list[Document]: List of Document objects representing the chunks of the PDF.
        """
        hash_check = make_hash_from_metadata(paper_meta)
        check_uniqueness = self.vectorstore.get(where={"hash": hash_check})
        if check_uniqueness.get("ids", 0):
            print("The document is already in the vectorstore.")
            return

        sections = preprocess_pdf(docs, paper_meta, hash_check)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts, metadatas = [], []
        for sec in sections:
            texts.append(sec["content"])
            metadatas.append(sec["metadata"])
        chunks = splitter.create_documents(texts, metadatas=metadatas)
        self.vectorstore.add_documents(chunks)
        return 
    
    def create_graph(self) -> CompiledStateGraph:
        """
        Creates a graph of execution, including retrieving, llm and structured output.

        Returns: CompiledStateGraph - runnable graph
        TODO: if we want to reuse some parts of the graph, we can create a class instead of a function
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_rag),
                ("human", "{question}"),
            ]
        )

        def retrieve(state: State, config: RunnableConfig):
            number_of_docs = config["configurable"].get("chunk_nums", 4)
            retrieved_docs = self.vectorstore.similarity_search(state["question"], number_of_docs)
            return {"context": retrieved_docs}


        def generate(state: State):
            formatted_docs = format_docs_with_id(state["context"])
            # docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": formatted_docs})
            structured_llm = self.llm.with_structured_output(QuotedAnswer)
            response = structured_llm.invoke(messages)
            return {"answer": response}


        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        
        return graph