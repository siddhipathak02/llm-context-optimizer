from services.pdf_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.relevance import score_chunks
from services.llm_client import ask_llm
from algorithms.models import Chunk
from algorithms.greedy import greedy_optimize
from algorithms.dynamic_programming import dynamic_programming_optimize
from algorithms.submodular import submodular_greedy_optimize

PDF_PATH = "test_docs/dbms_test.pdf"
QUESTION = "Explain normalization and functional dependencies in databases"
TOKEN_BUDGET = 1500

def main():
    print(f"Extracting text from {PDF_PATH} ...")
    full_text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(full_text)} characters")

    raw_chunks = chunk_text(full_text, words_per_chunk=150)
    print(f"Split into {len(raw_chunks)} chunks")

    print(f"\nScoring relevance against question: \"{QUESTION}\"")
    scores = score_chunks(QUESTION, raw_chunks)

    chunks = [
        Chunk(
            id=i + 1,
            token_cost=len(raw_chunks[i].split()),
            relevance_score=round(scores[i], 3),
            text=raw_chunks[i],
        )
        for i in range(len(raw_chunks))
    ]

    print(f"\nToken Budget: {TOKEN_BUDGET}")
    print(f"{'Algorithm':<24}{'Tokens':>10}{'Relevance':>14}{'Chunks':>10}")
    print("-" * 60)

    algo_results = {}
    for optimize in [greedy_optimize, dynamic_programming_optimize, submodular_greedy_optimize]:
        result = optimize(chunks, TOKEN_BUDGET)
        algo_results[result.algorithm_name] = result
        print(
            f"{result.algorithm_name:<24}"
            f"{result.total_tokens:>10}"
            f"{result.total_relevance:>14.3f}"
            f"{len(result.selected_chunks):>10}"
        )

    # ---- The actual payoff: optimized vs. full-document baseline ----
    full_context_words = len(full_text.split())
    optimized = algo_results["Submodular Greedy"]
    optimized_context = "\n\n".join(c.text for c in optimized.selected_chunks)
    optimized_words = sum(c.token_cost for c in optimized.selected_chunks)

    print("\n" + "=" * 60)
    print("FULL-CONTEXT BASELINE vs. OPTIMIZED (Submodular Greedy)")
    print("=" * 60)
    print(f"Full document:      {full_context_words} words sent to LLM")
    print(f"Optimized selection: {optimized_words} words sent to LLM")
    print(f"Token reduction: {100 * (1 - optimized_words / full_context_words):.1f}%")

    print("\n--- Answer using FULL document as context ---")
    full_answer = ask_llm(QUESTION, full_text)
    print(full_answer)

    print("\n--- Answer using OPTIMIZED (Submodular) selection as context ---")
    optimized_answer = ask_llm(QUESTION, optimized_context)
    print(optimized_answer)

if __name__ == "__main__":
    main()