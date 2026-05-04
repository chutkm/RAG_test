import argparse
import os
import numpy as np
import faiss

from huggingface_hub import InferenceClient
from pypdf import PdfReader
from docx import Document

from dotenv import load_dotenv

load_dotenv()


from config import (
    EMBEDDING_CONFIG,
    RETRIEVER_CONFIG,
    RERANKER_CONFIG,
    LLM_CONFIG,
)

# ---------------------------
# HF Client
# ---------------------------
def get_client(provider):
    return InferenceClient(
        provider=provider,
        api_key=os.environ["HF_TOKEN"],
    )


# ---------------------------
# Document loading
# ---------------------------
def load_document(path):
    if not os.path.exists(path):
        raise ValueError(f"Файл не найден: {path}")

    ext = path.lower().split(".")[-1]

    if ext == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == "pdf":
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    elif ext == "docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")


# ---------------------------
# Chunking
# ---------------------------
def split_text(text, chunk_size=512, overlap=128):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ---------------------------
# Embeddings (FIXED)
# ---------------------------
def embed_texts(client, texts, is_query=False):
    prefix = "query: " if is_query else "passage: "
    inputs = [prefix + t for t in texts]

    embeddings = client.feature_extraction(
        inputs,
        model=EMBEDDING_CONFIG["model"],
    )

    vectors = []

    for emb in embeddings:
        emb = np.array(emb)

        # CASE 1: already vector
        if emb.ndim == 1:
            vec = emb

        # CASE 2: token embeddings
        elif emb.ndim == 2:
            vec = emb.mean(axis=0)

        else:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")

        vectors.append(vec)

    vectors = np.array(vectors)

    # safety check
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # normalize (cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)

    return vectors


# ---------------------------
# FAISS
# ---------------------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(query, embed_client, index, chunks):
    q_emb = embed_texts(embed_client, [query], is_query=True)

    scores, indices = index.search(q_emb, RETRIEVER_CONFIG["top_k"])

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= RETRIEVER_CONFIG["score_threshold"]:
            results.append((chunks[idx], float(score)))

    return results


# ---------------------------
# Reranker
# ---------------------------
def rerank(query, candidates):
    if not RERANKER_CONFIG["enabled"] or not candidates:
        return candidates

    embed_client = get_client(EMBEDDING_CONFIG["provider"])

    # embedding запроса
    q_emb = embed_texts(embed_client, [query], is_query=True)[0]

    scored = []

    for text, _ in candidates:
        doc_emb = embed_texts(embed_client, [text])[0]

        
        score = float(np.dot(q_emb, doc_emb))

        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # top-k
    top = scored[:RERANKER_CONFIG["top_k"]]

    return top
# ---------------------------
# LLM
# ---------------------------
def generate_answer(question, contexts):
    if not contexts:
        return "Ответ не найден в документе"

    context_text = "\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": "Отвечай строго по контексту. Если ответа нет — скажи: 'Ответ не найден в документе'."
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context_text}\n\nВопрос: {question}"
        }
    ]

    client = get_client(LLM_CONFIG["provider"])

    response = client.chat_completion(
        messages=messages,
        model=LLM_CONFIG["model"],
        max_tokens=LLM_CONFIG["max_tokens"],
    )

    return response.choices[0].message["content"].strip()


# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print("Loading document...")
    text = load_document(args.document)

    print("Chunking...")
    chunks = split_text(text)

    print("Embedding via HF API...")
    embed_client = get_client(EMBEDDING_CONFIG["provider"])
    chunk_embeddings = embed_texts(embed_client, chunks)

    print("Building FAISS index...")
    index = build_index(chunk_embeddings)

    print("Retrieving...")
    retrieved = retrieve(args.question, embed_client, index, chunks)
    print(f"Retrieved: {len(retrieved)}")

    print("Reranking...")
    reranked = rerank(args.question, retrieved)

    contexts = [text for text, _ in reranked]

    print("\n=== ВОПРОС ===")
    print(args.question)

    answer = generate_answer(args.question, contexts)

    print("\n=== ОТВЕТ ===")
    print(answer)

    print("\n=== ИСТОЧНИКИ ===")
    for i, (chunk, score) in enumerate(reranked, 1):
        print(f"{i}) score={score:.4f}")
        print(chunk[:300])
        print()


if __name__ == "__main__":
    main()