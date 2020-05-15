from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class RF():
    def __init__(self):
        # self.rf = RandomForestClassifier(
        #     n_estimators = 500,
        #     criterion = 'gini',
        #     max_depth = 10,
        #     min_samples_split = 20,
        #     min_samples_leaf = 10,
        #     min_weight_fraction_leaf = 0.0,
        #     max_features ='auto',
        #     max_leaf_nodes = None,
        #     min_impurity_decrease = 0.0,
        #     min_impurity_split = None,
        #     bootstrap = True,
        #     oob_score = False,
        #     n_jobs = -1,
        #     random_state = None,
        #     verbose = 0,
        #     warm_start = False,
        #     class_weight = None)
        self.svc = SVC()


    def fit(self, X, Y):
        # self.rf.fit(X, Y)
        self.svc.fit(X, Y)

    def predict(self, X):
        # return self.rf.predict_proba(X)
        pred =  self.svc.predict(X)
        return pred




