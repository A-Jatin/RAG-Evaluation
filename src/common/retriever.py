from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configuration import CHUNK_SIZE, CHUNK_OVERLAP


class ChromaRetriever:

    def __init__(self, embeddings, docs):
        self.embeddings = embeddings
        self.docs = docs

    def create_vectorstore(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(self.docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        return vectorstore
