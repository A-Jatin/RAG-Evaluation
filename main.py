import os

from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.common.retriever import ChromaRetriever
from src.rag.factory import RAG_Factory
from src.evaluation.evaluation import RagasEvaluation
from configuration import OPENAI_KEY, MODEL_NAME, RAG_PROMPT, EVAL_DATASETS

os.environ["OPENAI_API_KEY"] = OPENAI_KEY


def evaluate_query(docs, query):
    llm = ChatOpenAI(model=MODEL_NAME)
    vectorstore = ChromaRetriever(docs=docs, embeddings=OpenAIEmbeddings()).create_vectorstore()
    retriever = vectorstore.as_retriever()
    results = {}
    for rag_type in RAG_Factory.SUPPORTED_RAG_TYPES.keys():
        rag = RAG_Factory(rag_type).build()
        results[rag_type] = rag(retriever, llm, RAG_PROMPT)(query)
    return results


def evaluate_rag():
    evaluations = []
    for dataset_name, dataset_split in EVAL_DATASETS:
        dataset = load_dataset(dataset_name, trust_remote_code=True)

        vectorstore = ChromaRetriever(docs=[context for context in dataset[dataset_split]["contexts"] if context],
                                      embeddings=OpenAIEmbeddings()).create_vectorstore()
        retriever = vectorstore.as_retriever()
        for rag_type in RAG_Factory.SUPPORTED_RAG_TYPES.keys():
            rag = RAG_Factory(rag_type).build()
            llm = ChatOpenAI(model=MODEL_NAME)
            dataset_with_results = []
            for row in dataset[dataset_split]:
                row["answer"] = rag(retriever, llm, RAG_PROMPT)(row["question"])
                dataset_with_results.append(row)
            evaluation = RagasEvaluation(dataset_with_results, llm).evaluate()
            evaluations.append({"dataset_name": dataset_name, "rag_type": rag_type, "evaluation": evaluation})
    return evaluations


if __name__ == "__main__":
    evaluations = evaluate_rag()
    print(evaluations)
