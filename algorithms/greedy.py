import time
from typing import List
from .models import Chunk, OptimizationResult

def greedy_optimize(chunks: List[Chunk], token_budget: int) -> OptimizationResult:
    """Select chunks by highest relevance-per-token ratio.

    This is fast but is not guaranteed to be optimal for 0/1 knapsack.
    Time: O(n log n), Space: O(n).
    """
    start = time.perf_counter()

    ranked = sorted(
        chunks,
        key=lambda c: c.relevance_score / c.token_cost if c.token_cost > 0 else float("inf"),
        reverse=True,
    )

    selected = []
    total_tokens = 0
    total_relevance = 0.0

    for chunk in ranked:
        if total_tokens + chunk.token_cost <= token_budget:
            selected.append(chunk)
            total_tokens += chunk.token_cost
            total_relevance += chunk.relevance_score

    elapsed = (time.perf_counter() - start) * 1000
    return OptimizationResult(
        "Greedy", selected, total_tokens, total_relevance, elapsed
    )
