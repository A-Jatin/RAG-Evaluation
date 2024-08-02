# RAG with langchain, openai and pinecone

from src.common.rag import UniversalRAG


class NaiveRAG(UniversalRAG):

    def __init__(self, retriever, llm, rag_prompt):
        super().__init__(retriever, llm, rag_prompt)
