from benchmark import generate_dataset, run_benchmark, TOPIC_VARIANTS

def topic_of(text: str) -> str:
    for topic, variants in TOPIC_VARIANTS.items():
        if text in variants:
            return topic
    return "unknown"

def main():
    token_budget = 1200
    chunks = generate_dataset(n=20, seed=42)
    results = run_benchmark(chunks, token_budget)

    print("=" * 78)
    print("LLM CONTEXT OPTIMIZER - DAA PROTOTYPE")
    print("=" * 78)
    print(f"Token Budget: {token_budget}")
    print(f"Number of Chunks: {len(chunks)}")
    print()

    print(
        f"{'Algorithm':<24}"
        f"{'Tokens':>10}"
        f"{'Relevance':>14}"
        f"{'Time(ms)':>14}"
        f"{'Chunks':>10}"
    )
    print("-" * 78)

    for result in results:
        print(
            f"{result.algorithm_name:<24}"
            f"{result.total_tokens:>10}"
            f"{result.total_relevance:>14.3f}"
            f"{result.execution_time_ms:>14.3f}"
            f"{len(result.selected_chunks):>10}"
        )

    print("-" * 78)
    print("\nSelected chunk IDs:")
    for result in results:
        ids = [c.id for c in result.selected_chunks]
        print(f"{result.algorithm_name:<24}: {ids}")

    print("\nDistinct topics covered (this is the real redundancy story):")
    for result in results:
        topics = sorted(set(topic_of(c.text) for c in result.selected_chunks))
        print(f"{result.algorithm_name:<24}: {len(topics)}/5  {topics}")

    dp = next(r for r in results if r.algorithm_name == "Dynamic Programming")
    knapsack_style = [r for r in results if r.algorithm_name != "Submodular Greedy"]
    print("\nValidation:")
    print(f"All solutions within budget: "
          f"{all(r.total_tokens <= token_budget for r in results)}")
    print(
        f"DP relevance >= every other knapsack-style result: "
        f"{all(dp.total_relevance + 1e-9 >= r.total_relevance for r in knapsack_style)}"
    )
    print(
        "\nNote: Submodular Greedy's 'Relevance' is net value AFTER a redundancy "
        "penalty, so it is not directly comparable to the other algorithms' raw "
        "relevance sums. Compare it by topic coverage above instead."
    )

if __name__ == "__main__":
    main()