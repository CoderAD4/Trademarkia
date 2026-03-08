import faiss
import numpy as np
class SemanticCache:
    def __init__(self, dim, threshold=0.85, k=5):
        self.threshold = threshold
        self.k = k
        self.index = faiss.IndexFlatIP(dim)
        self.queries = []
        self.results = []
        self.clusters = []
        self.hit_count = 0
        self.miss_count = 0
    def lookup(self, embedding, cluster):
        if len(self.queries) == 0:
            self.miss_count += 1
            return None
        embedding = np.array(embedding).astype("float32")
        D, I = self.index.search(embedding, min(self.k, len(self.queries)))
        best_match = None
        best_score = 0
        for rank in range(len(I[0])):
            idx = I[0][rank]
            similarity = float(D[0][rank])
            if self.clusters[idx] != cluster:
                continue
            if similarity > self.threshold and similarity > best_score:
                best_score = similarity
                best_match = idx
        if best_match is not None:
            self.hit_count += 1
            return {
                "cache_hit": True,
                "matched_query": self.queries[best_match],
                "similarity_score": best_score,
                "result": self.results[best_match]
            }
        self.miss_count += 1
        return None
    def store(self, query, embedding, result, cluster):
        embedding = np.array(embedding).astype("float32")
        self.index.add(embedding)
        self.queries.append(query)
        self.results.append(result)
        self.clusters.append(cluster)
    def stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = 0 if total == 0 else self.hit_count / total
        return {
            "total_entries": len(self.queries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 3)
        }
    def clear(self):
        self.index.reset()
        self.queries = []
        self.results = []
        self.clusters = []
        self.hit_count = 0
        self.miss_count = 0