from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Chunk:
    id: int
    token_cost: int
    relevance_score: float
    text: str

@dataclass
class OptimizationResult:
    algorithm_name: str
    selected_chunks: List[Chunk]
    total_tokens: int
    total_relevance: float
    execution_time_ms: float
