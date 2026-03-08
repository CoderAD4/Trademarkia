import numpy as np
from data.loader import load_dataset
from embeddings.embedder import Embedder
from clustering.reducer import Reducer
from clustering.gmm_cluster import ClusterModel
def build_clusters():
    print("Loading dataset...")
    docs = load_dataset()
    print("Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.encode(docs)
    print("Reducing dimensions with UMAP...")
    reducer = Reducer()
    reduced = reducer.fit_transform(embeddings)
    print("Training Gaussian Mixture clustering...")
    cluster_model = ClusterModel(n_clusters=20)
    cluster_model.fit(reduced)
    probs = cluster_model.predict_proba(reduced)
    print("Cluster probability matrix shape:", probs.shape)
    return cluster_model, probs
if __name__ == "__main__":
    build_clusters()