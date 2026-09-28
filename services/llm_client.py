import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = "openai/gpt-oss-120b"

def ask_llm(question: str, context: str) -> str:
    """Send a question plus supporting context to the LLM and return its answer."""
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content