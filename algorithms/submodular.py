import time
from typing import List, Dict
from .models import Chunk, OptimizationResult
from services.embeddings import get_embedding, cosine_similarity


def submodular_greedy_optimize(
    chunks: List[Chunk],
    token_budget: int,
    redundancy_weight: float = 0.5,
) -> OptimizationResult:
    """Select chunks to maximize relevance while penalizing redundancy.

    Unlike plain Greedy/DP, this does NOT treat each chunk's value as fixed
    and independent. Instead, at each step, a chunk's marginal value is:

        marginal_value = relevance - redundancy_weight * (similarity to most similar already-selected chunk)

    This means a highly relevant chunk that just repeats information we
    already selected contributes LESS than a moderately relevant chunk that
    covers new ground. This makes the selection problem submodular
    (diminishing returns), not additive like 0/1 Knapsack.

    Greedy is a natural fit here because for submodular maximization under a
    knapsack constraint, greedy-by-marginal-gain-per-cost has a proven
    (1 - 1/e) approximation guarantee (~63% of optimal), unlike plain
    knapsack greedy, which has no such guarantee.

    Time: O(n^2) in the worst case (each of n steps rescans remaining chunks
    and compares against selected ones). Space: O(n) for cached embeddings.
    """
    start = time.perf_counter()

    embeddings: Dict[int, object] = {c.id: get_embedding(c.text) for c in chunks}

    selected: List[Chunk] = []
    remaining = list(chunks)
    total_tokens = 0
    total_net_value = 0.0

    while remaining:
        best_chunk = None
        best_ratio = -float("inf")
        best_marginal = 0.0

        for chunk in remaining:
            if total_tokens + chunk.token_cost > token_budget:
                continue

            if selected:
                max_sim = max(
                    cosine_similarity(embeddings[chunk.id], embeddings[s.id])
                    for s in selected
                )
            else:
                max_sim = 0.0

            marginal_value = chunk.relevance_score - redundancy_weight * max_sim
            ratio = marginal_value / chunk.token_cost if chunk.token_cost > 0 else float("inf")

            if ratio > best_ratio:
                best_ratio = ratio
                best_chunk = chunk
                best_marginal = marginal_value

        if best_chunk is None:
            break

        selected.append(best_chunk)
        remaining.remove(best_chunk)
        total_tokens += best_chunk.token_cost
        total_net_value += best_marginal

    elapsed = (time.perf_counter() - start) * 1000
    return OptimizationResult(
        "Submodular Greedy", selected, total_tokens, total_net_value, elapsed
    )