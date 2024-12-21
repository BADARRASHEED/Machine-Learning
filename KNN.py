import numpy as np
from collections import Counter


# Global function for calculating Euclidean Distance
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # This function will be used to predict the class of the test data
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    # Helper function to predict a single sample
    def _predict(self, x):
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the closest k
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Vote for the most common label
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
