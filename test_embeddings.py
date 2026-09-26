from services.embeddings import get_embedding, cosine_similarity

v1 = get_embedding("The cat sat on the mat")
v2 = get_embedding("A feline rested on the rug")
v3 = get_embedding("The stock market crashed today")

print("Similar sentences:", cosine_similarity(v1, v2))
print("Unrelated sentences:", cosine_similarity(v1, v3))
