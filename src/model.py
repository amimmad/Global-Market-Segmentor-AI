from sklearn.cluster import KMeans


class ClusterModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )

    def fit_predict(self, data):
        return self.model.fit_predict(data)

    def get_inertia(self):
        return self.model.inertia_

    def get_centroids(self):
        return self.model.cluster_centers_
