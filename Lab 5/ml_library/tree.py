import numpy as np


class Node:
    def __init__(self, feature = None, treshold  = None, left = None, right = None, value = None):
        self.feature = feature
        self.treshold = treshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None


class CART:
    def __init__(self, max_depth = 10, min_samples=2, type = 'classifier'):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.type = type
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(np.array(X), np.array(y))
    
    def predict(self, X):
        return np.array([self.travers_tree(x, self.tree) for x in np.array(X)])
    
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return np.sum([p*(1-p) for p in ps])
    
    def _mse(self, y):
        return np.mean((y-y.mean())**2)
    
    def most_common(self, y):
        if len(y) == 0:
            return 0 
        if self.type == 'classifier':
            labels, counts = np.unique(y, return_counts=True)
            return labels[np.argmax(counts)]
        else:  
            return np.mean(y)
    
    def _cost_function(self, left_df, right_df, method):
        total_df_size = left_df.size + right_df.size
        p_left_df = left_df.size / total_df_size
        p_right_df = right_df.size / total_df_size
        J_left = method(left_df)
        J_right = method(right_df)
        J = p_left_df*J_left + p_right_df*J_right

        return J
    
    def _count_labels(self, X_feature, treshold, y):
        left_indexes = np.argwhere(X_feature <= treshold).flatten()
        right_indexes = np.argwhere(X_feature > treshold).flatten()

        return y[left_indexes], y[right_indexes]
    
    def best_split(self, X, y):
        best_feature, best_treshold = None, None
        best_gain = -np.inf
        method = self._gini if self.type == 'classifier' else self._mse
        parent_impurity = method(y)

        for i in range(X.shape[1]):
            tresholds = np.unique(X[:, i])
            if len(tresholds) < 2: 
                continue
            for j in range(1, len(tresholds)):
                current_tresh = tresholds[j]
                previos_tresh = tresholds[j-1]
                treshold = (current_tresh + previos_tresh) / 2
                left_labels, right_labels = self._count_labels(X[:, i], treshold, y)
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue
                current_J = self._cost_function(left_labels, right_labels, method)
                gain = parent_impurity - current_J
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_treshold = treshold
        
        return best_feature, best_treshold
    
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if n_labels ==1 or depth >= self.max_depth or n_samples <= self.min_samples:
            return Node(value= self.most_common(y))
        
        best_feature , best_treshold = self.best_split(X,y)
        left_indexes = np.argwhere(X[:, best_feature]<=best_treshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature]> best_treshold).flatten()

        left = self.build_tree(X[left_indexes, :], y[left_indexes], depth+1)
        right = self.build_tree(X[right_indexes, :], y[right_indexes], depth+1)

        return Node(best_feature, best_treshold, left, right)
    def travers_tree(self, x, tree):
        if tree.is_leaf_node():
            return tree.value
        if x[tree.feature]< tree.treshold:
            return self.travers_tree(x, tree.left)
        
        return self.travers_tree(x, tree.right)