import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-3.8-flash"

def ask_llm(question: str, context: str) -> str:
    """Send a question plus supporting context to the LLM and return its answer."""
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = _client.models.generate_content(model=_MODEL, contents=prompt)
    return response.text