from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache
embedder = Embedder()
dim = embedder.encode(["test"]).shape[1]
cache = SemanticCache(dim)
query = "space shuttle launch"
embedding = embedder.encode([query])
result = ["Document about NASA launch"]
cache.store(query, embedding, result)
lookup = cache.lookup(embedding)
print(lookup)