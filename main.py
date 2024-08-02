import os

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
    evaluations = {}
    for rag_type in RAG_Factory.SUPPORTED_RAG_TYPES.keys():
        rag = RAG_Factory(rag_type).build()
        results[rag_type] = rag(retriever, llm, RAG_PROMPT)(query)
    return results


def evaluate_rag():
    from datasets import load_dataset

    for dataset_name in EVAL_DATASETS:
        dataset = load_dataset(dataset_name)
        vectorstore = ChromaRetriever(docs=dataset["train"]["document"], embeddings=OpenAIEmbeddings()).\
            create_vectorstore()
        retriever = vectorstore.as_retriever()
        for rag_type in RAG_Factory.SUPPORTED_RAG_TYPES.keys():
            rag = RAG_Factory(rag_type).build()
            llm = ChatOpenAI(model=MODEL_NAME)
            results = rag(retriever, llm, RAG_PROMPT)(dataset["eval"]["query"])
            evaluation = RagasEvaluation(results, llm).evaluate()
            print(f"Dataset: {dataset_name}, RAG Type: {rag_type}, Evaluation: {evaluation}")
            print(results)

