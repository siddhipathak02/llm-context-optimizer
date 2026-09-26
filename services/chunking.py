from typing import List

def chunk_text(text: str, words_per_chunk: int = 150, min_chunk_words: int = 30) -> List[str]:
    """Split text into chunks of roughly `words_per_chunk` words each.

    If the final leftover chunk is too small (< min_chunk_words), merge it
    into the previous chunk instead of leaving it as a tiny, unstable chunk
    (a very small token_cost distorts relevance-per-token ratios).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    if len(chunks) > 1 and len(chunks[-1].split()) < min_chunk_words:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + " " + last

    return chunks