from services.pipeline import build_chunks_from_pdf, run_all_algorithms
from services.llm_client import ask_llm
from services.pdf_parser import extract_text_from_pdf

PDF_PATH = "test_docs/python_tutorial.pdf"
QUESTION = "How do you check if students passed or failed and calculate their average marks in Python?"
TOKEN_BUDGET = 700

def main():
    print(f"Extracting and scoring chunks from {PDF_PATH} ...")
    chunks = build_chunks_from_pdf(PDF_PATH, QUESTION)
    print(f"Got {len(chunks)} chunks")

    results = run_all_algorithms(chunks, TOKEN_BUDGET)

    print(f"\nToken Budget: {TOKEN_BUDGET}")
    print(f"{'Algorithm':<24}{'Tokens':>10}{'Relevance':>14}{'Chunks':>10}")
    print("-" * 60)
    for name, result in results.items():
        print(
            f"{name:<24}"
            f"{result.total_tokens:>10}"
            f"{result.total_relevance:>14.3f}"
            f"{len(result.selected_chunks):>10}"
        )
        print("\nSelected chunk IDs:")
    for name, result in results.items():
        print(f"{name:<24}: {[c.id for c in result.selected_chunks]}")
    from services.evaluation import total_relevance, redundancy_score
    print("\nTrade-off: relevance captured vs. internal redundancy (lower redundancy = more diverse):")
    print(f"{'Algorithm':<24}{'Relevance':>14}{'Redundancy':>14}")
    for name, result in results.items():
        rel = total_relevance(result.selected_chunks)
        red = redundancy_score(result.selected_chunks)
        print(f"{name:<24}{rel:>14.3f}{red:>14.3f}")
        


    full_text = extract_text_from_pdf(PDF_PATH)
    full_words = len(full_text.split())
    optimized = results["Submodular Greedy"]
    optimized_context = "\n\n".join(c.text for c in optimized.selected_chunks)
    optimized_words = sum(c.token_cost for c in optimized.selected_chunks)

    print("\n" + "=" * 60)
    print("FULL-CONTEXT BASELINE vs. OPTIMIZED (Submodular Greedy)")
    print("=" * 60)
    print(f"Full document:       {full_words} words")
    print(f"Optimized selection: {optimized_words} words")
    print(f"Token reduction: {100 * (1 - optimized_words / full_words):.1f}%")

    print("\n--- Answer using FULL document as context ---")
    try:
        print(ask_llm(QUESTION, full_text))
    except Exception as e:
        print(f"[FAILED: full document exceeds the LLM's rate/size limit]\n{e}")

    print("\n--- Answer using OPTIMIZED (Submodular) selection as context ---")
    print(ask_llm(QUESTION, optimized_context))

if __name__ == "__main__":
    main()