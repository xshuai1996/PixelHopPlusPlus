import numpy as np
from saab import Saab
from skimage.util import view_as_windows


class cwSaab():
    """ Channel-wise Saab transform. Proposed in paper \"PixelHop++: A Small Successive-Subspace-Learning-Based
    (SSL-based) Model for Image Classification\""""
    def __init__(self, energyTH=0.01, SaabArgs=None, neighborArgs=None, poolingArg=None):
        """energyTH: to control if will be decompose in the next layer. """
        self.par = {}
        self.depth = len(SaabArgs)
        self.energyTH = energyTH
        self.SaabArgs = SaabArgs
        self.neighborArgs = neighborArgs
        self.poolingArg = poolingArg
        self.Energy = []
        self.trained = False
        self.split = False

    def SaabTransform(self, X, saab, train, layer):
        SaabArg = self.SaabArgs[layer]
        X = self.Neighbor(X, layer)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
        if train == True:
            saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
            saab.fit(X)
        # use batch to avoid memory error
        batch_size = int(X.shape[0]/5)
        x1 = saab.transform(X[: 1 * batch_size])
        x2 = saab.transform(X[1 * batch_size: 2 * batch_size])
        x3 = saab.transform(X[2 * batch_size: 3 * batch_size])
        x4 = saab.transform(X[3 * batch_size: 4 * batch_size])
        x5 = saab.transform(X[4 * batch_size:])
        del X
        X = np.concatenate((x1, x2, x3, x4, x5), axis=0)
        del x1, x2, x3, x4, x5
        X = X.reshape(S)
        X = self.Pooling(X, layer)
        return saab, X

    def cwSaab_1_layer(self, X, train):
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer' + str(0)]
        transformed, eng = [], []
        if train == True:
            saab, transformed = self.SaabTransform(X, saab=None, train=True, layer=0)
            saab_cur.append(saab)
            eng.append(saab.Energy)
        else:
            _, transformed = self.SaabTransform(X, saab=saab_cur[0], train=False, layer=0)

        if train == True:
            self.par['Layer' + str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return transformed

    def cwSaab_n_layer(self, X, train, layer):
        output, eng_cur, ct, pidx = [], [], -1, 0
        S = list(X.shape)
        S[-1] = 1
        X = np.moveaxis(X, -1, 0)               # channel 1st to ease splitting. Will be converted back by reshape in the future
        saab_prev = self.par['Layer'+str(layer-1)]
        if train == True:
            saab_cur = []
        else:
            saab_cur = self.par['Layer'+str(layer)]
        for i in range(len(saab_prev)):     # in cw Saab the previous layer would be split
            for j in range(saab_prev[i].Energy.shape[0]):
                ct += 1
                if saab_prev[i].Energy[j] < self.energyTH:
                    continue
                self.split = True
                X_tmp = X[ct].reshape(S).copy()      # ct = 0, 1, ...
                if train == True:
                    saab, out_tmp = self.SaabTransform(X_tmp, saab=None, train=True, layer=layer)
                    saab.Energy *= saab_prev[i].Energy[j]
                    saab_cur.append(saab)
                    eng_cur.append(saab.Energy) 
                else:
                    _, out_tmp = self.SaabTransform(X_tmp, saab=saab_cur[pidx], train=False, layer=layer)
                    pidx += 1
                output.append(out_tmp)
        if self.split == True:
            output = np.concatenate(output, axis=-1)
            if train == True:
                self.par['Layer'+str(layer)] = saab_cur
                self.Energy.append(np.concatenate(eng_cur, axis=0))
        return output
    
    def fit(self, X):
        X = self.cwSaab_1_layer(X, train=True)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=True, layer=i)
            if self.split == False:
                self.depth = i
                print("       <WARNING> Cannot further split, actual depth: %s" % str(i))
                break
        self.trained = True

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        output = []
        X = self.cwSaab_1_layer(X, train=False)
        output.append(X)
        for i in range(1, self.depth):
            X = self.cwSaab_n_layer(X, train=False, layer=i)
            output.append(X)
        return output

    def Neighbor(self, X, idx):
        kernel = self.neighborArgs[idx]['kernel']
        stride = self.neighborArgs[idx]['stride']
        X = view_as_windows(X, kernel, stride)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], -1))
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], -1))
        return X

    def Pooling(self, X, idx):
        win_size = self.poolingArg[idx]['win']
        win_params = (1, win_size, win_size, X.shape[3])
        if self.poolingArg[idx]['pad'] is True and X.shape[1] % win_size != 0:
            num_wins = X.shape[1] // win_size + 1
            new_X = np.full((X.shape[0], num_wins * win_size, num_wins * win_size, X.shape[3]), -np.inf)
            new_X[:, :X.shape[1], :X.shape[2], :] = X
            X = new_X
        X = view_as_windows(X, win_params, win_params)
        X = np.max(X, axis=(5, 6), keepdims=True)
        X = np.squeeze(X, axis=(3, 4, 5, 6))
        return X

