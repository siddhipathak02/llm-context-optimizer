import random
from typing import List
from algorithms.models import Chunk, OptimizationResult
from algorithms.greedy import greedy_optimize
from algorithms.dynamic_programming import dynamic_programming_optimize
from algorithms.approximation import approximation_optimize
from algorithms.randomized import randomized_optimize
from algorithms.submodular import submodular_greedy_optimize
from services.relevance import score_chunks

# A small pool of topics, each with a few paraphrased variants, so the
# synthetic dataset contains realistic redundancy (like a real document
# repeating the same idea in different sections).
TOPIC_VARIANTS = {
    "normalization": [
        "Normalization removes redundancy in database tables by splitting them properly.",
        "Database normalization reduces redundant data by decomposing tables correctly.",
        "Organizing tables to eliminate duplicate data is called normalization.",
    ],
    "functional_dependency": [
        "A functional dependency means one column's value determines another column's value.",
        "In a functional dependency, one attribute's value is determined by another.",
    ],
    "indexing": [
        "Indexes speed up query performance by avoiding full table scans.",
        "Database indexes allow faster lookups instead of scanning every row.",
    ],
    "transactions": [
        "A transaction is a sequence of operations executed as a single unit.",
        "ACID properties ensure transactions are processed reliably.",
    ],
    "joins": [
        "A join combines rows from two or more tables based on a related column.",
        "SQL joins let you query data spread across multiple related tables.",
    ],
}


def generate_dataset(n: int = 20, seed: int = 42) -> List[Chunk]:
    """Generate reproducible synthetic document chunks with realistic topic overlap."""
    rng = random.Random(seed)
    all_texts = [text for variants in TOPIC_VARIANTS.values() for text in variants]
    texts = [rng.choice(all_texts) for _ in range(n)]

    question = "Explain normalization and functional dependencies in databases"
    scores = score_chunks(question, texts)

    return [
        Chunk(
            id=i + 1,
            token_cost=rng.randint(80, 300),
            relevance_score=round(scores[i], 3),
            text=texts[i],
        )
        for i in range(n)
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
        submodular_greedy_optimize(chunks, token_budget),
    ]

    for result in results:
        if result.total_tokens > token_budget:
            raise ValueError(
                f"{result.algorithm_name} exceeded budget: "
                f"{result.total_tokens} > {token_budget}"
            )

    return results