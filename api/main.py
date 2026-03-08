import os
import threading
import webbrowser
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from data.loader import load_dataset
from embeddings.embedder import Embedder
from vector_db.faiss_index import VectorDB
from clustering.reducer import Reducer
from clustering.gmm_cluster import ClusterModel
from cache.semantic_cache import SemanticCache
app = FastAPI()
class QueryRequest(BaseModel):
    query: str
# Global system objects
docs = None
embedder = None
vectordb = None
reducer = None
cluster_model = None
cache = None
# ---------------------------
# SYSTEM INITIALIZATION
# ---------------------------
def initialize_system():
    global docs, embedder, vectordb, reducer, cluster_model, cache
    print("Loading dataset...")
    docs = load_dataset()
    print("Initializing embedder...")
    embedder = Embedder()
    print("Generating document embeddings...")
    embeddings = embedder.encode(docs)
    dim = embeddings.shape[1]
    print("Building vector database...")
    vectordb = VectorDB(dim)
    vectordb.add_embeddings(embeddings, docs)
    print("Reducing embeddings for clustering...")
    reducer = Reducer()
    reduced_embeddings = reducer.fit_transform(embeddings)
    print("Training Gaussian Mixture clustering...")
    cluster_model = ClusterModel(n_clusters=20)
    cluster_model.fit(reduced_embeddings)
    print("Initializing semantic cache...")
    cache = SemanticCache(dim)
    print("System ready!")
# ---------------------------
# STARTUP EVENT
# ---------------------------
@app.on_event("startup")
def startup_event():
    initialize_system()
    print("Opening browser...")
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8000/docs")
    threading.Timer(1.5, open_browser).start()
# ---------------------------
# ROOT ENDPOINT
# ---------------------------
@app.get("/")
def root():
    return {"message": "Semantic Search API Running"}
# ---------------------------
# QUERY ENDPOINT
# ---------------------------
@app.post("/query")
def query_endpoint(request: QueryRequest):
    query = request.query
    query_embedding = embedder.encode([query])
    query_reduced = reducer.reducer.transform(query_embedding)
    cluster_probs = cluster_model.predict_proba(query_reduced)
    dominant_cluster = int(np.argmax(cluster_probs))
    cached = cache.lookup(query_embedding, dominant_cluster)
    if cached is not None:
        cached["query"] = query
        cached["dominant_cluster"] = dominant_cluster
        return cached
    results = vectordb.search(query_embedding, k=5)
    cache.store(query, query_embedding, results, dominant_cluster)
    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster
    }
# ---------------------------
# CACHE STATS
# ---------------------------
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()
# ---------------------------
# CLEAR CACHE
# ---------------------------
@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}