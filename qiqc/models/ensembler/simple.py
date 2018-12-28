from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from qiqc.model_selection import classification_metrics


class AverageEnsembler(object):

    def __init__(self, models, results, device, batchsize_train,
                 batchsize_valid):
        self.models = models
        self.results = results
        self.device = device
        self.batchsize_train = batchsize_train
        self.batchsize_valid = batchsize_valid

    def fit(self, X, t, sampling=0.1):
        train_X = X.to(self.device)
        iterator = DataLoader(
            train_X, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='ensemble', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(batch))
        ys = [np.concatenate(_ys) for _ys in ys.values()]
        y = np.array(ys).mean(axis=0)
        metrics = classification_metrics(y, t)
        self.threshold = metrics['threshold']
        return y, metrics

    def predict_proba(self, X):
        pred_X = X.to(self.device)
        iterator = DataLoader(
            pred_X, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='submit', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(batch))
        ys = [np.concatenate(_ys) for _ys in ys.values()]
        y = np.array(ys).mean(axis=0)
        return y

    def predict(self, X):
        y = self.predict_proba(X)
        return (y > self.threshold).astype('i')
