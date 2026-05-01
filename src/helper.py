from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
import os
from huggingface_hub import InferenceClient


from huggingface_hub import InferenceClient

class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self._client = InferenceClient(token=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = self._client.feature_extraction(text, model=self.model_name)
            if hasattr(emb, "tolist"):
                embeddings.append(emb.tolist())
            else:
                embeddings.append(emb)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        emb = self._client.feature_extraction(text, model=self.model_name)
        if hasattr(emb, "tolist"):
            return emb.tolist()
        return emb


def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    print("DEBUG: Using custom HuggingFaceInferenceEmbeddings class")
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY is not set in environment variables")
    
    return HuggingFaceInferenceEmbeddings(
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
