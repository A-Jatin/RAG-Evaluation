from langchain_chroma import Chroma

from src.common.utils import format_texts


class ChromaRetriever:

    def __init__(self, embeddings, docs):
        self.embeddings = embeddings
        self.docs = docs

    def create_vectorstore(self):
        formatted_texts = [format_texts(doc) for doc in self.docs]
        vectorstore = Chroma.from_texts(texts=formatted_texts, embedding=self.embeddings)
        return vectorstore
