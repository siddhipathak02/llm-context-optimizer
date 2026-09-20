import random
import time
from typing import List, Optional
from .models import Chunk, OptimizationResult

def randomized_optimize(
    chunks: List[Chunk],
    token_budget: int,
    iterations: int = 300,
    seed: Optional[int] = 42,
) -> OptimizationResult:
    """Randomized heuristic that repeatedly constructs candidate solutions.

    At each iteration, chunks are considered in a randomized order biased
    toward useful relevance/token ratios. The best valid solution is kept.

    Typical time: O(iterations * n log n), Space: O(n).
    """
    start = time.perf_counter()
    rng = random.Random(seed)

    if not chunks or token_budget <= 0:
        return OptimizationResult(
            "Randomized", [], 0, 0.0,
            (time.perf_counter() - start) * 1000
        )

    best = ([], 0, 0.0)

    for _ in range(iterations):
        order = chunks[:]
        # Add random noise to the ratio to vary candidate order.
        order.sort(
            key=lambda c: (
                c.relevance_score / c.token_cost if c.token_cost > 0 else float("inf")
            ) * rng.uniform(0.85, 1.15),
            reverse=True,
        )

        selected, used, score = [], 0, 0.0
        for chunk in order:
            # Randomized acceptance creates different feasible candidates.
            ratio = chunk.relevance_score / chunk.token_cost if chunk.token_cost else 0
            acceptance = min(1.0, 0.35 + ratio / max(1.0, max(
                c.relevance_score / c.token_cost if c.token_cost else 0 for c in chunks
            )))
            if used + chunk.token_cost <= token_budget and rng.random() < acceptance:
                selected.append(chunk)
                used += chunk.token_cost
                score += chunk.relevance_score

        if score > best[2]:
            best = (selected, used, score)

    elapsed = (time.perf_counter() - start) * 1000
    return OptimizationResult(
        "Randomized", best[0], best[1], best[2], elapsed
    )
