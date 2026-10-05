from typing import List, Dict
from algorithms.models import Chunk, OptimizationResult
from algorithms.greedy import greedy_optimize
from algorithms.dynamic_programming import dynamic_programming_optimize
from algorithms.approximation import approximation_optimize
from algorithms.randomized import randomized_optimize
from algorithms.submodular import submodular_greedy_optimize
from services.pdf_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.relevance import score_chunks

from functools import partial

ALGORITHMS = {
    "Greedy": greedy_optimize,
    "Dynamic Programming": dynamic_programming_optimize,
    "Approximation": approximation_optimize,
    "Randomized": randomized_optimize,
    "Submodular Greedy": partial(submodular_greedy_optimize, redundancy_weight=1.0),
}
def build_chunks_from_pdf(pdf_path: str, question: str, words_per_chunk: int = 150) -> List[Chunk]:
    """Extract, split, and score chunks from a PDF, ready for optimization."""
    full_text = extract_text_from_pdf(pdf_path)
    raw_chunks = chunk_text(full_text, words_per_chunk=words_per_chunk)
    scores = score_chunks(question, raw_chunks)

    return [
        Chunk(
            id=i + 1,
            token_cost=len(raw_chunks[i].split()),
            relevance_score=round(scores[i], 3),
            text=raw_chunks[i],
        )
        for i in range(len(raw_chunks))
    ]

def run_all_algorithms(chunks: List[Chunk], token_budget: int) -> Dict[str, OptimizationResult]:
    """Run every registered algorithm on the same chunks and budget."""
    return {name: fn(chunks, token_budget) for name, fn in ALGORITHMS.items()}