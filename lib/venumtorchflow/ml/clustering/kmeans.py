import numpy as np


class KMeans:
    """
    kmeans algorithm object
    """

    def __init__(self, data, k, iters, tol, *, init="normal"):
        """
        initialize kmeans object

        parameters
        ----------
        data: np.array
            data to be clustered
        k: int
            number of clusters
        iters: int
            number of iterations
        tol: float
            tolerance level
        init: str
            method to initialize centroids
        """
        self.data_length, self.data_dim = data.shape
        self.data = data
        self.k = k
        self.iters = iters
        self.tol = tol
        self.init = init

    def train(self):
        """ Call to train the model """
        self.centroids = {i: self.data[i] for i in range(self.k)}

        for i in range(self.iters):
            self.classifications = {i: [] for i in range(self.k)}

            for features in self.data:
                distances = [np.linalg.norm(features - self.centroids[centroid])
                             for centroid in self.centroids]

                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = \
                    np.average(self.classifications[classification], axis=0)

                optimized = True

                for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]

                    if np.sum((current_centroid - original_centroid) / original_centroid * 100) > self.tol:
                        optimized = False

                if optimized:
                    break

    def predict(self, data):
        """
        predict new data using trained model

        parameters:
        -----------
        data: np.array
            data to be classified
        """
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
