from services.llm_client import ask_llm

answer = ask_llm(
    "What is normalization?",
    "Normalization is the process of organizing database tables to reduce redundancy."
)
print(answer)