# 2020.04.09
import numpy as np 
from cwSaab import cwSaab

class Pixelhop2(cwSaab):
    def __init__(self, TH1=0.001, TH2=0.0001, SaabArgs=None, neighborArgs=None, poolingArg=None, concatArg=None):
        super().__init__(energyTH=TH1, SaabArgs=SaabArgs, neighborArgs=neighborArgs, poolingArg=poolingArg)
        self.TH1 = TH1
        self.TH2 = TH2
        self.idx = []
        self.concatArg = concatArg

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