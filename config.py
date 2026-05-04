EMBEDDING_CONFIG = {
    "model": "intfloat/multilingual-e5-large",
    "provider": "hf-inference",
    "batch_size": 16,
}

RETRIEVER_CONFIG = {
    "top_k": 10,
    "score_threshold": 0.5,
}

RERANKER_CONFIG = {
    "enabled": True,
    "model": "BAAI/bge-reranker-large",
    "provider": "hf-inference",
    "top_k": 5,
}

LLM_CONFIG = {
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "provider": "novita",
    "max_tokens": 512,
}