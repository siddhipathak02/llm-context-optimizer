from services.pdf_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.relevance import score_chunks
from algorithms.models import Chunk
from algorithms.greedy import greedy_optimize
from algorithms.dynamic_programming import dynamic_programming_optimize
from algorithms.submodular import submodular_greedy_optimize

PDF_PATH = "test_docs/dbms_test.pdf"
QUESTION = "Explain normalization and functional dependencies in databases"
TOKEN_BUDGET = 1500

def main():
    print(f"Extracting text from {PDF_PATH} ...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text)} characters")

    raw_chunks = chunk_text(text, words_per_chunk=150)
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

    for optimize in [greedy_optimize, dynamic_programming_optimize, submodular_greedy_optimize]:
        result = optimize(chunks, TOKEN_BUDGET)
        print(
            f"{result.algorithm_name:<24}"
            f"{result.total_tokens:>10}"
            f"{result.total_relevance:>14.3f}"
            f"{len(result.selected_chunks):>10}"
        )

    print("\n--- Chunks selected by Submodular Greedy ---")
    submod_result = submodular_greedy_optimize(chunks, TOKEN_BUDGET)
    for c in submod_result.selected_chunks:
        preview = c.text[:80].replace("\n", " ")
        print(f"[id {c.id}, {c.token_cost} tok, rel {c.relevance_score:.3f}] {preview}...")

if __name__ == "__main__":
    main()