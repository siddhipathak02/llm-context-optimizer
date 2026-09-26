# step1_chunker.py

# This is our "document" for now — just a plain string.
# Later this will come from a real PDF, but let's keep it simple first.
document = """
Artificial intelligence is transforming how we build software.
Large language models can understand and generate human-like text.
However, they have a limited context window, meaning they can only
read a certain amount of text at once. This creates a problem when
we have huge documents that don't fit into that window. We need a
way to pick the most useful pieces of text and discard the rest.
This project explores different algorithms to solve that problem.
"""

def chunk_text(text, chunk_size):
    """Split text into pieces of chunk_size characters each."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Try it out
chunks = chunk_text(document, chunk_size=100)

for i, c in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(c)
    print()