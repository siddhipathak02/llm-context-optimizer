from itertools import combinations
from typing import List
from algorithms.models import Chunk
from services.embeddings import get_embedding, cosine_similarity


def total_relevance(selected: List[Chunk]) -> float:
    """Raw relevance captured by a selection — directly comparable across all algorithms."""
    return sum(c.relevance_score for c in selected)


def redundancy_score(selected: List[Chunk]) -> float:
    """Average pairwise similarity among selected chunks. Lower = more diverse/less overlap.

    Applied the SAME way to every algorithm's output after the fact, so we can
    compare how much internal overlap each algorithm's selection contains,
    regardless of how that algorithm reasoned while choosing.
    """
    if len(selected) < 2:
        return 0.0

    embeddings = {c.id: get_embedding(c.text) for c in selected}
    pair_similarities = [
        cosine_similarity(embeddings[a.id], embeddings[b.id])
        for a, b in combinations(selected, 2)
    ]
    return sum(pair_similarities) / len(pair_similarities)