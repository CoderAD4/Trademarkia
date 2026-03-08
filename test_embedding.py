from embeddings.embedder import Embedder
embedder = Embedder()
vector = embedder.encode(["hello world"])
print("Embedding shape:", vector.shape)