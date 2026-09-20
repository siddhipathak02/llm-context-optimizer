import random
from typing import List
from algorithms.models import Chunk, OptimizationResult
from algorithms.greedy import greedy_optimize
from algorithms.dynamic_programming import dynamic_programming_optimize
from algorithms.approximation import approximation_optimize
from algorithms.randomized import randomized_optimize

def generate_dataset(n: int = 20, seed: int = 42) -> List[Chunk]:
    """Generate reproducible synthetic document chunks."""
    rng = random.Random(seed)
    return [
        Chunk(
            id=i,
            token_cost=rng.randint(80, 300),
            relevance_score=round(rng.uniform(0.2, 1.0), 3),
            text=f"Synthetic document chunk {i}",
        )
        for i in range(1, n + 1)
    ]

def run_benchmark(
    chunks: List[Chunk], token_budget: int, random_seed: int = 42
) -> List[OptimizationResult]:
    """Run every algorithm on the exact same dataset and budget."""
    results = [
        greedy_optimize(chunks, token_budget),
        dynamic_programming_optimize(chunks, token_budget),
        approximation_optimize(chunks, token_budget),
        randomized_optimize(chunks, token_budget, seed=random_seed),
    ]

    for result in results:
        if result.total_tokens > token_budget:
            raise ValueError(
                f"{result.algorithm_name} exceeded budget: "
                f"{result.total_tokens} > {token_budget}"
            )

    return results
