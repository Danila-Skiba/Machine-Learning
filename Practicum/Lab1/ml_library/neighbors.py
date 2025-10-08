import numpy as np




class DistanceMetric:
    def compute(self, X_train, x_test_i):
        pass


class EuclideanDistance(DistanceMetric):
    def compute(self, X_train, x_test_i):
        return np.sqrt(np.sum((X_train - x_test_i) ** 2, axis=1))
    
class ManhattanDistance(DistanceMetric):
    def compute(self, X_train, x_test_i):
        return np.sum(np.abs(X_train - x_test_i), axis=1)
    

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, metric= None):
        self.n_neighbors = n_neighbors
        self.metric = metric if metric is not None else EuclideanDistance()

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

    def _compute_distances(self, x_test_i):
        return self.metric.compute(self.X_train, x_test_i)

    def _make_prediction(self, x_test_i):
        distances = self._compute_distances(x_test_i)   
        k_nearest_indexes = np.argsort(distances)[:self.n_neighbors]
        targets = self.y_train[k_nearest_indexes]  

        return np.bincount(targets).argmax()

    def predict(self, X_test):
        return np.array([self._make_prediction(x) for x in X_test])
    
