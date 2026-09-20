import time
from typing import List
from .models import Chunk, OptimizationResult

def _ratio_greedy(chunks: List[Chunk], token_budget: int):
    ranked = sorted(
        chunks,
        key=lambda c: c.relevance_score / c.token_cost if c.token_cost > 0 else float("inf"),
        reverse=True,
    )
    selected, used, score = [], 0, 0.0
    for chunk in ranked:
        if used + chunk.token_cost <= token_budget:
            selected.append(chunk)
            used += chunk.token_cost
            score += chunk.relevance_score
    return selected, used, score

def approximation_optimize(
    chunks: List[Chunk], token_budget: int
) -> OptimizationResult:
    """A simple approximation heuristic.

    Compares ratio-based greedy with the best single feasible chunk.
    It is not guaranteed to equal the optimum.

    Time: O(n log n), Space: O(n).
    """
    start = time.perf_counter()

    greedy_selected, greedy_tokens, greedy_score = _ratio_greedy(
        chunks, token_budget
    )

    feasible = [c for c in chunks if c.token_cost <= token_budget]
    best_single = max(feasible, key=lambda c: c.relevance_score, default=None)

    single_score = best_single.relevance_score if best_single else 0.0
    if single_score > greedy_score:
        selected = [best_single]
        total_tokens = best_single.token_cost
        total_relevance = single_score
    else:
        selected = greedy_selected
        total_tokens = greedy_tokens
        total_relevance = greedy_score

    elapsed = (time.perf_counter() - start) * 1000
    return OptimizationResult(
        "Approximation", selected, total_tokens, total_relevance, elapsed
    )
