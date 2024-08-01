# RAG with langchain, openai and pinecone

from src.common.rag import UniversalRAG


def naive_rag(llm, vectorstore, rag_prompt, query):
    # Retrieve and generate using the relevant snippets of the blog.

    retriever = vectorstore.as_retriever()
    rag = UniversalRAG(retriever, llm, rag_prompt)
    return rag(query)

