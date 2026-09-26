from algorithms.models import Chunk
from algorithms.greedy import greedy_optimize
from algorithms.submodular import submodular_greedy_optimize

# Two near-duplicate high-relevance chunks, plus some distinct lower-relevance ones
chunks = [
    Chunk(id=1, token_cost=300, relevance_score=0.95,
          text="Normalization removes redundancy in database tables by splitting them properly."),
    Chunk(id=2, token_cost=300, relevance_score=0.93,
          text="Database normalization reduces redundant data by decomposing tables correctly."),
    Chunk(id=3, token_cost=300, relevance_score=0.60,
          text="A functional dependency means one column's value determines another column's value."),
    Chunk(id=4, token_cost=300, relevance_score=0.55,
          text="Indexes speed up query performance by avoiding full table scans."),
]

budget = 700  # only room for ~2 chunks

for result in [greedy_optimize(chunks, budget), submodular_greedy_optimize(chunks, budget)]:
    print(f"\n{result.algorithm_name}")
    print(f"  Selected: {[c.id for c in result.selected_chunks]}")
    print(f"  Tokens: {result.total_tokens}")
    print(f"  Value: {result.total_relevance:.3f}")