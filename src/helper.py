from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from typing import List
from langchain.schema import Document


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
    # FastEmbed: ultra-lightweight local embeddings (~50MB RAM, no API key needed)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings
