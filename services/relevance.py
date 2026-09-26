from services.embeddings import get_embedding, cosine_similarity

def score_chunks(question: str, chunk_texts: list[str]) -> list[float]:
    """Return a relevance score (0 to 1ish) for each chunk, based on similarity to the question."""
    question_vec = get_embedding(question)
    scores = []
    for text in chunk_texts:
        chunk_vec = get_embedding(text)
        scores.append(cosine_similarity(question_vec, chunk_vec))
    return scores