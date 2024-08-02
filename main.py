import os

from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.common.retriever import ChromaRetriever
from src.rag.factory import RAG_Factory
from configuration import OPENAI_KEY, MODEL_NAME, RAG_PROMPT

os.environ["OPENAI_API_KEY"] = OPENAI_KEY


def run_query(docs, query):
    llm = ChatOpenAI(model=MODEL_NAME)
    vectorstore = ChromaRetriever(docs=docs, embeddings=OpenAIEmbeddings()).create_vectorstore()
    retriever = vectorstore.as_retriever()
    results = {}
    for rag_type in RAG_Factory.SUPPORTED_RAG_TYPES.keys():
        rag = RAG_Factory(rag_type).build()
        results[rag_type] = rag(retriever, llm, RAG_PROMPT)(query)
    return results
