from sklearn.mixture import GaussianMixture
class ClusterModel:
    def __init__(self, n_clusters=20):
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )
    def fit(self, embeddings):
        self.model.fit(embeddings)
    def predict_proba(self, embeddings):
        return self.model.predict_proba(embeddings)