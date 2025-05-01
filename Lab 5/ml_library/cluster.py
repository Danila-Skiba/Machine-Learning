import numpy as np
import random

import pandas as pd

class KMeansCustom:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centers = []
        self.cluster_centers_ = None
        self.clusters = [[] for _ in range(n_clusters)]
        self.num_features = None
        self.labels_ = None
    
    def init_centers(self, points):
        min_values = np.min(points, axis=0)
        max_values = np.max(points, axis=0)
        for _ in range(self.n_clusters):
            center = [random.uniform(min_values[i], max_values[i]) for i in range(points.shape[1])]
            self.centers.append(center)

    @staticmethod
    def euclidean_distance(point_cur, points):
        dists = []
        for point in points:
            dist = sum([(point_cur[i] - point[i])**2 for i in range(len(point))])
            dists.append(dist)
        return dists
    
    def update_centers(self):
        for cluster_i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                cluster_mean_value = [0] * self.num_features
                for point in cluster:
                    for i in range(self.num_features):
                        cluster_mean_value[i] += point[i]
                for i in range(self.num_features):
                    cluster_mean_value[i] /= len(cluster)
                self.centers[cluster_i] = cluster_mean_value

    @staticmethod
    def get_item_cluster(distances):
        min_item_i = 0
        for index, dist in enumerate(distances):
            if dist < distances[min_item_i]:
                min_item_i = index
        return min_item_i
    
    def fit(self, points, epochs=100):
        labels = []
        if isinstance(points, pd.DataFrame):
            points = points.values
        self.num_features = points.shape[1]
        self.init_centers(points)
        for _ in range(epochs):
            self.clusters = [[] for _ in range(self.n_clusters)]
            labels = []
            for point in points:
                dists = self.euclidean_distance(point, self.centers)
                item_cluster = self.get_item_cluster(dists)
                self.clusters[item_cluster].append(point)
                labels.append(item_cluster)
            self.update_centers()
        
        self.labels_ = labels
        self.cluster_centers_ = np.array(self.centers)
    