from services.pipeline import build_chunks_from_pdf, run_all_algorithms
from services.llm_client import ask_llm
from services.pdf_parser import extract_text_from_pdf

PDF_PATH = "test_docs/dbms_test.pdf"
QUESTION = "Explain normalization and functional dependencies in databases"
TOKEN_BUDGET = 300

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
    print(ask_llm(QUESTION, full_text))

    print("\n--- Answer using OPTIMIZED (Submodular) selection as context ---")
    print(ask_llm(QUESTION, optimized_context))

if __name__ == "__main__":
    main()