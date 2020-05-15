import numpy as np
from cwSaab import cwSaab


class Pixelhop2(cwSaab):
    def __init__(self, TH1=0.001, TH2=0.0001, SaabArgs=None, neighborArgs=None, poolingArg=None):
        """ TH1: energy threshold to control if will be decompose in next layer in c/w Saab transform
            TH2: to determine if will be kept for feature selection and decision."""
        super().__init__(energyTH=TH1, SaabArgs=SaabArgs, neighborArgs=neighborArgs, poolingArg=poolingArg)
        self.TH1 = TH1
        self.TH2 = TH2
        self.idx = []

    def select_(self, X):
        for i in range(self.depth):
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
        return X

    def fit(self, X):
        super().fit(X)

    def transform(self, X):
        X = super().transform(X)
        X = self.select_(X)
        return X