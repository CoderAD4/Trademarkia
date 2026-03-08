# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight **semantic search engine** built on top of the **20 Newsgroups dataset**. Instead of relying on keyword matching, the system uses **vector embeddings and similarity search** to retrieve documents that are semantically related to a user query.

The system also includes a **semantic cache layer** that detects when similar queries have already been processed and reuses previously computed results to avoid redundant computation. The entire system is exposed through a **FastAPI service**, making it easy to interact with through API requests.

The goal of this project is to demonstrate how modern NLP techniques such as **sentence embeddings, vector databases, clustering, and semantic caching** can be combined to build an efficient semantic search pipeline.

---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains around **20,000 documents across 20 discussion categories**.

Some example categories include:

- Technology
- Sports
- Politics
- Religion
- Science

Since the dataset contains a lot of noisy information (email headers, signatures, quoted replies), a cleaning step is applied before processing.

---

# Data Cleaning

The dataset contains several elements that do not contribute to semantic meaning. The preprocessing step removes these to improve embedding quality.

Cleaning steps include:

- Removing email headers, footers, and quoted replies
- Removing email addresses and URLs
- Removing numeric noise
- Normalizing whitespace
- Removing extremely short documents
- Truncating each document to **2000 characters** to avoid long inputs slowing down embedding generation

This produces a cleaner dataset that is more suitable for semantic embeddings.

---

# System Architecture

The system is built as a modular pipeline:

```
Dataset
   ↓
Data Cleaning
   ↓
Sentence Embeddings
   ↓
Vector Database (FAISS)
   ↓
UMAP Dimensionality Reduction
   ↓
Gaussian Mixture Fuzzy Clustering
   ↓
Semantic Cache
   ↓
FastAPI Service
```

Each component plays a role in enabling efficient semantic search.

---

# Embeddings

Documents are converted into vector representations using the **Sentence Transformers model**:

```
all-MiniLM-L6-v2
```

This model produces **384-dimensional embeddings** that capture semantic meaning.

Advantages of this model:

- Lightweight and fast
- Good semantic performance
- Suitable for CPU environments

These embeddings allow semantically similar documents to appear close to each other in vector space.

---

# Vector Database

The system uses **FAISS (Facebook AI Similarity Search)** to store and search document embeddings.

FAISS enables fast **nearest-neighbor search**, allowing the system to retrieve semantically similar documents efficiently.

When a query is submitted:

1. The query is converted into an embedding
2. FAISS finds the closest document vectors
3. The most similar documents are returned

---

# Dimensionality Reduction

Before clustering, embeddings are reduced using **UMAP**.

UMAP helps:

- Reduce dimensionality
- Preserve semantic structure
- Improve clustering performance

Embeddings are reduced from **384 dimensions to 50 dimensions**.

---

# Fuzzy Clustering

Clustering is performed using a **Gaussian Mixture Model (GMM)**.

Unlike traditional clustering methods that assign each document to a single cluster, GMM produces **probability distributions over clusters**.

Example:

```
Document A
Cluster 3 → 0.52
Cluster 7 → 0.31
Cluster 12 → 0.17
```

This allows documents to belong to multiple clusters with varying probabilities, which is why the clustering approach is considered **fuzzy clustering**.

---

# Semantic Cache

The semantic cache avoids recomputing results for similar queries.

Instead of caching exact query strings, the system:

1. Converts queries into embeddings
2. Searches for similar past queries
3. Reuses cached results if similarity exceeds a threshold

For example:

```
"space shuttle launch"
"nasa rocket launch"
"space mission nasa"
```

These queries are different syntactically but semantically similar, so the cache can reuse results.

Each cache entry stores:

- Query text
- Query embedding
- Search results
- Cluster information

---

# FastAPI Service

The system is exposed through a **FastAPI server**.

### Query Endpoint

```
POST /query
```

Example request:

```json
{
  "query": "space shuttle launch"
}
```

Example response:

```json
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "result": [...],
  "dominant_cluster": 3
}
```

---

### Cache Statistics

```
GET /cache/stats
```

Returns:

- Total cache entries
- Cache hit count
- Cache miss count
- Hit rate

---

### Clear Cache

```
DELETE /cache
```

Removes all cached entries.

---

# Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Generate system artifacts

```
python setup_system.py
```

This step generates embeddings, vector indexes, and clustering models.

### Start the API

```
uvicorn api.main:app --reload
```

The API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

# Technologies Used

- Python
- FastAPI
- Sentence Transformers
- FAISS
- UMAP
- Scikit-learn
- NumPy

---

# Key Features

- Semantic search using transformer embeddings
- Vector similarity search with FAISS
- Fuzzy clustering using Gaussian Mixture Models
- Cluster-aware semantic caching
- REST API interface via FastAPI

---

# Future Improvements

Possible improvements include:

- Cluster visualization
- Improved cache eviction strategies
- Query ranking improvements
- Adding a frontend interface
- Containerization with Docker

---

# Conclusion

This project demonstrates how modern NLP techniques can be combined to build an efficient semantic search system. By combining embeddings, vector databases, clustering, and semantic caching, the system can retrieve relevant information while minimizing redundant computation.

The architecture is modular, making it easy to extend or improve individual components in the future.
