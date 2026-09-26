from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Convert a piece of text into a numeric vector representing its meaning."""
    return _model.encode(text)

def cosine_similarity(vec1, vec2) -> float:
    """Return how similar two vectors are, from -1 (opposite) to 1 (identical)."""
    return float(cos_sim(vec1, vec2))