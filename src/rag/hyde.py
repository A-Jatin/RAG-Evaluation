from src.common.rag import UniversalRAG


class HyDE(UniversalRAG):
    def __init__(self, retriever, llm, rag_prompt):
        super().__init__(retriever, llm, rag_prompt)

    def generate_hypothetical_answer(self, query):
        from langchain.prompts import ChatPromptTemplate
        # Creating the Prompt Template
        template = """For the given question try to generate a hypothetical answer\
        Only generate the answer and nothing else:
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        query = prompt.format(question=query)

        return self.llm.invoke(query).content

    def retrieval(self, query):
        hypothetical_answer = self.generate_hypothetical_answer(query)
        similar_docs = super().retrieval(hypothetical_answer)
        return similar_docs
