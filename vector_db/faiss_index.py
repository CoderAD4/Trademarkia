import faiss
import numpy as np
class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 100
        self.documents = []
    def add_embeddings(self, embeddings, docs):
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(docs)
    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])
        return results