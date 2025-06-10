import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "deepseek-r1:8b",
        llm_temperature: float = 0.7,
        vector_store: FAISS = None  # ⬅️ Accept FAISS object directly
    ):
        """
        Initializes the ChatbotManager with embedding model, LLM, and FAISS store.

        Args:
            model_name (str): HuggingFace model for embeddings.
            device (str): 'cpu' or 'cuda'.
            encode_kwargs (dict): Encoding options.
            llm_model (str): Ollama model name.
            llm_temperature (float): Temperature for generation.
            vector_store (FAISS): Pre-initialized FAISS vector store.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

        # Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # LLM (e.g., Ollama)
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature
        )

        # Prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If the context does not provide enough information, answer the question based on your own knowledge, but state that it comes from your knoledge and not present in the document.
If you don't know the answer, say that you don't know.

Context: {context}
Question: {question}

Answer:
"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # Vector store (must be set via setter if not passed)
        self.vector_store = vector_store

        if self.vector_store is not None:
            self._initialize_qa()

    def _initialize_qa(self):
        """
        Initializes RetrievalQA from the current vector store.
        """
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt},
            verbose=False
        )

    def set_vector_store(self, vector_store: FAISS):
        """
        Sets the FAISS vector store and initializes QA pipeline.

        Args:
            vector_store (FAISS): A pre-loaded FAISS store.
        """
        self.vector_store = vector_store
        self._initialize_qa()

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.
        It hides <thinking> in the UI, but keeps it in logs.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's visible response (without <thinking>), logs contain full.
        """
        try:
            # First try retrieval-augmented answer
            response = self.qa.run(query)

            # If response is empty or fallback needed
            if not response.strip() or "don't know" in response.lower():
                response = self.llm(query)

            # Ensure it's a string (some LangChain chains might return other objects)
            if not isinstance(response, str):
                response = str(response)

            return response  # full response (including <thinking>) to be logged
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
