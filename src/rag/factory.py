from src.rag.naive_rag import naive_rag
from src.rag.hyde import HyDE


class RAG_Factory:

    SUPPORTED_RAG_TYPES = {
        "naive": naive_rag,
        "hyde": HyDE
    }

    def __init__(self, rag_type):
        self.rag_type = rag_type

    def build(self):

        if self.rag_type in self.SUPPORTED_RAG_TYPES:
            return self.SUPPORTED_RAG_TYPES[self.rag_type]
        else:
            raise ValueError(f"RAG type {self.rag_type} not supported.")
