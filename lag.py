import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances


class LAG():
    """ Label Assisted Regression units. Proposed in paper \"pixelhop: a successive subspace learning (ssl) method for
    object recognition\". Run K-means inside each class and output the probability of each class."""
    def __init__(self, learner, num_clusters, alpha=5):
        """ num_clusters: array, # clusters inside each class
            alpha: param to control the influence of L2 distance"""
        assert (str(learner.__class__) == "<class 'llsr.LLSR'>"), "Currently only support <class 'llsr.LLSR'>!"
        self.learner = learner
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.clus_labels = []
        self.centroid = []
        self.num_class = []
        self.trained = False

    def compute_target_(self, X, Y, batch_size):
        Y = Y.reshape(-1)
        class_list = np.unique(Y)
        labels = np.zeros((X.shape[0]))
        self.clus_labels = np.zeros((np.sum(self.num_clusters),))
        self.centroid = np.zeros((np.sum(self.num_clusters), X.shape[1]))
        start = 0
        for i in range(len(class_list)):
            ID = class_list[i]
            feature_train = X[Y == ID]
            if batch_size == None:
                kmeans = KMeans(n_clusters=self.num_clusters[i], verbose=0, random_state=9, n_jobs=10).fit(
                    feature_train)
            else:
                kmeans = MiniBatchKMeans(n_clusters=self.num_clusters[i], verbose=0, batch_size=batch_size,
                                         n_init=5).fit(feature_train)
            labels[Y == ID] = kmeans.labels_ + start
            self.clus_labels[start:start + self.num_clusters[i]] = ID
            self.centroid[start:start + self.num_clusters[i]] = kmeans.cluster_centers_
            start += self.num_clusters[i]
        return labels.astype('int32')

    def fit(self, X, Y, batch_size=None):
        self.num_class = len(np.unique(Y))
        assert (len(self.num_clusters) >= self.num_class), "'len(num_cluster)' must larger than class number!"
        Yt = self.compute_target_(X, Y, batch_size=batch_size)
        Yt_onehot = np.zeros((Yt.shape[0], self.clus_labels.shape[0]))
        for i in range(Yt.shape[0]):
            gt = Y[i].copy()
            dis = euclidean_distances(X[i].reshape(1, -1), self.centroid[self.clus_labels == gt]).reshape(-1)
            dis = dis / (np.min(dis) + 1e-15)
            p_dis = np.exp(-dis * self.alpha)
            p_dis = p_dis / np.sum(p_dis)
            Yt_onehot[i, self.clus_labels == gt] = p_dis

        self.learner.fit(X, Yt_onehot)
        self.trained = True

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        return self.learner.predict_proba(X)

    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = self.transform(X)
        pred_labels = np.zeros((X.shape[0], self.num_class))
        for km_i in range(self.num_class):
            pred_labels[:, km_i] = np.sum(X[:, self.clus_labels == km_i], axis=1)
        pred_labels = pred_labels / np.sum(pred_labels, axis=1, keepdims=1)
        return pred_labels

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # def score(self, X, Y):
    #     assert (self.trained == True), "Must call fit first!"
    #     pred_labels = self.predict(X)
    #     idx = (pred_labels == Y.reshape(-1))
    #     return np.count_nonzero(idx) / Y.shape[0]
