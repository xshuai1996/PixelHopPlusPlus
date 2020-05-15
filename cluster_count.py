import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed


class K_means_transform():
    """ Use ideas from paper \"an analysis of single-layer networks in unsupervised feature learning\" to replace feature
    selection and LAG module in first layers to reduce running time and number of parameters. """
    def __init__(self, num_cluster=500, batch_size=5000):
        self.kmeans = MiniBatchKMeans(n_clusters=num_cluster, verbose=0, random_state=None, batch_size=batch_size)
        self.num_classes = num_cluster

    def batch_fit(self, X):
        self.shape_4D = X.shape
        X = X.reshape((-1, self.shape_4D[-1]))
        self.kmeans.partial_fit(X)

    def predict(self, X):
        X = X.reshape((-1, self.shape_4D[-1]))
        pred = self.kmeans.predict(X)
        pred = pred.reshape((self.shape_4D[0], self.shape_4D[1] * self.shape_4D[2]))
        cnt = np.zeros((self.shape_4D[0], self.num_classes))
        # due to memory limitation, change pooling to 4 quadrants to count
        def fill_in_cnt(class_id):
            for i in range(self.shape_4D[0]):
                cnt[i, class_id] = np.sum(pred[i] == class_id)

        Parallel(n_jobs=-1, require='sharedmem')(
            delayed(fill_in_cnt)(class_id)
            for class_id in range(self.num_classes))
        return cnt



