import numpy as np
import pickle
import faiss
from data.loader import load_dataset
from embeddings.embedder import Embedder
from clustering.reducer import Reducer
from clustering.gmm_cluster import ClusterModel
from vector_db.faiss_index import VectorDB
docs = load_dataset()
embedder = Embedder()
embeddings = embedder.encode(docs)
np.save("embeddings.npy", embeddings)
dim = embeddings.shape[1]
vectordb = VectorDB(dim)
vectordb.add_embeddings(embeddings, docs)
faiss.write_index(vectordb.index, "vector.index")
reducer = Reducer()
reduced = reducer.fit_transform(embeddings)
cluster_model = ClusterModel()
cluster_model.fit(reduced)
pickle.dump(cluster_model, open("cluster_model.pkl", "wb"))
pickle.dump(reducer, open("reducer.pkl", "wb"))
pickle.dump(docs, open("documents.pkl", "wb"))