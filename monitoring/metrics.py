import mlflow
import numpy as np
import time
import os
from itertools import combinations
from nltk.tokenize import word_tokenize


ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() in ("1", "true", "yes")

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two texts based on token sets.
    """
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    union = tokens1 | tokens2
    return float(len(tokens1 & tokens2) / len(union)) if union else 0.0


def result_set_diversity(embeddings: list[list[float]]) -> float:
    """
    Pairwise average cosine distance among embeddings (diversity measure).
    """
    if len(embeddings) < 2:
        return 0.0
    sims = [cosine_similarity(a, b) for a, b in combinations(embeddings, 2)]
    # diversity as average distance = 1 - average similarity
    return float(1 - np.mean(sims))


def overlap_diversity(texts: list[str]) -> float:
    """
    Pairwise average Jaccard similarity among texts (lower = more diverse).
    """
    if len(texts) < 2:
        return 0.0
    jacs = [jaccard_similarity(a, b) for a, b in combinations(texts, 2)]
    return float(np.mean(jacs))


def embedding_norm_stats(embeddings: list[list[float]]) -> tuple[float, float]:
    """
    Mean and std deviation of embedding norms.
    """
    norms = [np.linalg.norm(e) for e in embeddings]
    return float(np.mean(norms)), float(np.std(norms))


def session_coverage_ratio(session_ids: list[str]) -> float:
    """
    Fraction of unique sessions in the list.
    """
    if not session_ids:
        return 0.0
    return float(len(set(session_ids)) / len(session_ids))


def measure_retrieval_metrics(top_chunk, get_embedding) -> None:
    """
    Given the top item retrieval results, compute embeddings, similarities, texts,
    session IDs, and log all retrieval metrics to MLflow.

    Each run that calls this function should have already called:
        mlflow.set_experiment(...) 
        mlflow.start_run(...)
    so that these metrics are tied to a specific MLflow run.
    """

    if not ENABLE_MLFLOW:
        return

    if top_chunk == [] or top_chunk is None or top_chunk == '':
        mlflow.log_metric("retrieval/avg_similarity", 0.0)
        mlflow.log_metric("retrieval/latency_sec", float(time.time() - start_time))
        mlflow.log_metric("retrieval/result_set_diversity", 0.0)
        mlflow.log_metric("retrieval/overlap_diversity", 0.0)
        mlflow.log_metric("retrieval/embedding_norm_mean", 0.0)
        mlflow.log_metric("retrieval/embedding_norm_std", 0.0)
        mlflow.log_metric("retrieval/session_coverage_ratio", 0.0)
        return

    start_time = time.time()

    # Unpack exactly what from top_chunk
    sims = [sim for *_, sim in top_chunk]
    texts = [text for _, _, text, _ in top_chunk]
    session_ids = [session_id for *_, _, session_id in top_chunk]
    embeddings = [get_embedding(text) for _, _, text, _ in top_chunk]

    latency_sec = time.time() - start_time

    # 1) Average similarity
    avg_sim = float(np.mean(sims)) if sims else 0.0
    mlflow.log_metric("retrieval/avg_similarity", avg_sim)

    # 2) Latency
    mlflow.log_metric("retrieval/latency_sec", float(latency_sec))

    # 3) Result-set diversity (using embeddings)
    diversity_val = result_set_diversity(embeddings)
    mlflow.log_metric("retrieval/result_set_diversity", diversity_val)

    # 4) Overlap diversity (using texts)
    overlap_val = overlap_diversity(texts)
    mlflow.log_metric("retrieval/overlap_diversity", overlap_val)

    # 5) Embedding-norm statistics
    mean_norm, std_norm = embedding_norm_stats(embeddings)
    mlflow.log_metric("retrieval/embedding_norm_mean", float(mean_norm))
    mlflow.log_metric("retrieval/embedding_norm_std",  float(std_norm))

    # 6) Session coverage ratio (using session_ids)
    coverage_val = session_coverage_ratio(session_ids)
    mlflow.log_metric("retrieval/session_coverage_ratio", coverage_val)


def measure_generation_metrics(
    user_query: str,
    retrieved_chunks,
    answer: str,
    get_embedding
) -> None:
    """
    Given the user query, the list of retrieved chunks, and the generated answer,
    compute and log all generation-related metrics to MLflow.
    """

    if not ENABLE_MLFLOW:
        return

    # Compute latency
    start_time = time.time()

    latency_sec = time.time() - start_time
    mlflow.log_metric("generation/latency_sec", float(latency_sec))

    # Answer length
    answer_length = len(answer.split())
    mlflow.log_metric("generation/answer_length", answer_length)

    # Semantic similarity between query and answer
    q_emb = get_embedding(user_query)
    a_emb = get_embedding(answer)
    sim_qa = cosine_similarity(q_emb, a_emb)
    mlflow.log_metric("generation/semantic_similarity", float(sim_qa))

    # Combine chunk texts into one context string
    context = "\n\n".join(f"Chunk {i+1}: {chunk[2]}" for i, chunk in enumerate(retrieved_chunks))

    # Jaccard similarity between context and answer
    jacc = jaccard_similarity(context, answer)
    mlflow.log_metric("generation/context_answer_jaccard", float(jacc))

    # Self-coherence: average pairwise cosine similarity among sentence embeddings
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    sent_embs = [get_embedding(s) for s in sentences]
    if len(sent_embs) > 1:
        sims = [cosine_similarity(e1, e2) for e1, e2 in combinations(sent_embs, 2)]
        self_coh = float(np.mean(sims))
    else:
        self_coh = 0.0
    mlflow.log_metric("generation/self_coherence", self_coh)

    # Compression ratio: answer length / context length
    context_len = len(context.split())
    comp_ratio = float(answer_length / context_len) if context_len else 0.0
    mlflow.log_metric("generation/compression_ratio", comp_ratio)