import os
import numpy as np

from data.loader import load_dataset
from embeddings.embedder import Embedder
from vector_db.faiss_index import VectorDB


class SearchEngine:

    def __init__(self):

        print("Loading dataset...")
        self.docs = load_dataset()

        print("Initializing embedder...")
        self.embedder = Embedder()

        # Check if embeddings already exist
        if os.path.exists("embeddings.npy"):

            print("Loading saved embeddings...")
            embeddings = np.load("embeddings.npy")

        else:

            print("Generating embeddings (first run, may take a few minutes)...")
            embeddings = self.embedder.encode(self.docs)

            np.save("embeddings.npy", embeddings)

            print("Embeddings saved!")

        dim = embeddings.shape[1]

        print("Building vector database...")
        self.vectordb = VectorDB(dim)

        self.vectordb.add_embeddings(embeddings, self.docs)

        print("Search engine ready!")

    def search(self, query):

        query_embedding = self.embedder.encode([query])

        results = self.vectordb.search(query_embedding, k=5)

        return results