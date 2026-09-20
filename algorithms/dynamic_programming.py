import time
from typing import List
from .models import Chunk, OptimizationResult

def dynamic_programming_optimize(
    chunks: List[Chunk], token_budget: int
) -> OptimizationResult:
    """Solve the 0/1 knapsack formulation exactly.

    Relevance scores are scaled to integers so they can be represented
    in the DP table without floating-point indexing.

    Time: O(n * B), Space: O(n * B).
    """
    start = time.perf_counter()
    n = len(chunks)
    scale = 1000
    values = [max(0, int(round(c.relevance_score * scale))) for c in chunks]

    dp = [[0] * (token_budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost = chunks[i - 1].token_cost
        value = values[i - 1]
        for b in range(token_budget + 1):
            dp[i][b] = dp[i - 1][b]
            if cost <= b:
                candidate = dp[i - 1][b - cost] + value
                if candidate > dp[i][b]:
                    dp[i][b] = candidate

    selected = []
    b = token_budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            chunk = chunks[i - 1]
            selected.append(chunk)
            b -= chunk.token_cost

    selected.reverse()
    total_tokens = sum(c.token_cost for c in selected)
    total_relevance = sum(c.relevance_score for c in selected)

    elapsed = (time.perf_counter() - start) * 1000
    return OptimizationResult(
        "Dynamic Programming", selected, total_tokens, total_relevance, elapsed
    )
