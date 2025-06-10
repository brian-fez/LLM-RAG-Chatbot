import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS  # for production qdrant docker

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
    ):
        """
        Initializes the EmbeddingsManager with the specified model and FAISS backend.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        self.vector_store = None
        self.text_chunks = None

    def create_embeddings(self, pdf_path: str):
        """
        Processes the PDF, creates embeddings, and stores them in FAISS (in-memory).

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            str: Success message upon completion.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load and preprocess the document
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        # Create and store embeddings in FAISS (in memory)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.text_chunks = splits

        return "✅ Vector DB Successfully Created and Stored in FAISS (in-memory)!"

    def search(self, query: str, k: int = 3):
        """
        Search for top-k relevant chunks in the FAISS index.

        Args:
            query (str): The user query.
            k (int): Number of top results to return.

        Returns:
            List of relevant document chunks.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please create embeddings first.")

        return self.vector_store.similarity_search(query, k=k)
