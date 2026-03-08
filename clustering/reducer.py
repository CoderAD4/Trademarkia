import umap
class Reducer:
    def __init__(self):
        self.reducer = umap.UMAP(
            n_components=50,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine"
        )
    def fit_transform(self, embeddings):
        reduced = self.reducer.fit_transform(embeddings)
        return reduced