from data.loader import load_dataset
from embeddings.embedder import Embedder
from vector_db.faiss_index import VectorDB
def build_vector_database():
    print("Loading dataset...")
    docs = load_dataset()
    print("Total documents:", len(docs))
    embedder = Embedder()
    print("Generating embeddings...")
    embeddings = embedder.encode(docs)
    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)
    vectordb = VectorDB(dim)
    print("Adding embeddings to FAISS index...")
    vectordb.add_embeddings(embeddings, docs)
    print("Index built successfully!")
    return vectordb

if __name__ == "__main__":
    vectordb = build_vector_database()