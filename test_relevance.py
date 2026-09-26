from services.relevance import score_chunks

question = "Explain normalization and functional dependencies in databases"

chunks = [
    "Normalization is the process of organizing database tables to reduce redundancy.",
    "A functional dependency means one attribute determines another attribute's value.",
    "The stock market saw heavy volatility this week amid interest rate concerns.",
    "Python is a popular programming language used for data science and web development."
]

scores = score_chunks(question, chunks)
for text, score in zip(chunks, scores):
    print(f"{score:.3f}  {text}")