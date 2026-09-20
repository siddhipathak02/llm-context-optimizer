# LLM Context Optimizer

A DAA-based prototype for selecting the most useful text chunks under a fixed LLM token budget.

## Problem

An LLM cannot always receive every available document chunk because the context window is limited.

Each chunk has:

- `token_cost`: how many tokens it consumes
- `relevance_score`: how useful it is

Given a token budget `B`, the optimizer selects a subset that maximizes total relevance while satisfying:

`sum(token_cost) <= B`

This is formulated as a 0/1 Knapsack-style optimization problem.

## Algorithms

### Greedy
Ranks chunks by relevance per token and selects them while possible.

- Time: O(n log n)
- Space: O(n)
- Fast, but not guaranteed optimal for 0/1 knapsack.

### Dynamic Programming
Uses the classic 0/1 Knapsack DP formulation and reconstructs the selected subset.

- Time: O(nB)
- Space: O(nB)
- Produces the optimal solution for the discretized relevance values used here.

### Approximation
Compares ratio-based greedy with the best single feasible chunk.

- Time: O(n log n)
- Space: O(n)
- Heuristic/approximation; not guaranteed optimal.

### Randomized
Constructs many randomized candidate solutions and keeps the best valid solution found.

- Typical time: O(k n log n), where k is the number of iterations.
- Space: O(n)
- Repeated runs can vary unless a seed is fixed.

## Run

Python 3.10+ is recommended.

```bash
python main.py
```

No external packages are required for Phase 1.

## Current Scope

This phase uses synthetic chunks so that the DAA algorithms can be tested independently of AI/LLM infrastructure.

## Planned Future Work

1. PDF/document text extraction
2. Real text chunking
3. Relevance scoring for a user query
4. LLM integration
5. React visualization dashboard
6. Experimental benchmarking on larger datasets
7. Optional browser/Chrome extension
