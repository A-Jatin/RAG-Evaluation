from langchain import hub

RAG_PROMPT = hub.pull("rlm/rag-prompt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

OPENAI_KEY = ""
MODEL_NAME = "gpt-4o-mini"

EVAL_DATASETS = ["atitaarora/qdrant_docs_qna_ragas", "explodinggradients/amnesty_qa"]

SUPPORTED_RAG_TYPES = ["naive", "hyde"]
