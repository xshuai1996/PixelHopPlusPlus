from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA
import numpy as np
from joblib import Parallel, delayed


class LAFR():
    def __init__(self):
        """ Label assisted feature reduction. Used in module 2 of first layers. """
        # self.pca = IncrementalPCA(n_components=num_PCA_kernels)
        self.rf = RandomForestClassifier(
            n_estimators = 500,
            criterion = 'gini',
            max_depth = 15,
            min_samples_split = 2,
            min_samples_leaf = 50,
            min_weight_fraction_leaf = 0.0,
            max_features ='auto',
            max_leaf_nodes = None,
            min_impurity_decrease = 0.0,
            min_impurity_split = None,
            bootstrap = True,
            oob_score = False,
            n_jobs = -1,
            random_state = None,
            verbose = 0,
            warm_start = False,
            class_weight = None)


    def fit(self, X, Y):
        self.rf.fit(X, Y)

    def predict(self, X):
        return self.rf.predict_proba(X)




