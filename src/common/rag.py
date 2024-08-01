from langchain_core.output_parsers import StrOutputParser
from src.common.utils import format_docs


class UniversalRAG:

    def __init__(self, retriever, llm, rag_prompt):
        self.retriever = retriever
        self.llm = llm
        self.rag_prompt = rag_prompt

    def retrieval(self, query):
        similar_docs = self.retriever.get_relevant_documents(query)
        formatted_docs = format_docs(similar_docs)
        return formatted_docs

    def generation(self, query, formatted_docs):
        prompt = self.rag_prompt.format(context=formatted_docs, question=query)
        output = self.llm.invoke(prompt).content
        return StrOutputParser().parse(output)

    def __call__(self, query):
        formatted_docs = self.retrieval(query)
        return self.generation(query, formatted_docs)
