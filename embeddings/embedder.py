from sentence_transformers import SentenceTransformer
import torch
class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device
        )
    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings