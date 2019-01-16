from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from qiqc.model_selection import classification_metrics
from qiqc.models.ensembler.base import BaseEnsembler


class AverageEnsembler(BaseEnsembler):

    def __init__(self, config, models, results):
        self.config = config
        self.models = models
        self.results = results
        self.device = config['device']
        self.batchsize_train = config['batchsize']
        self.batchsize_valid = config['batchsize_valid']
        self.threshold_cv = np.array(
            [m.threshold for m in models]).mean()

    def fit(self, X, t, test_size=0.1):
        if not self.config['ensembler'].get('retrain_threshold'):
            self.threshold = self.threshold_cv
            return
        n_tests = int(len(X) * test_size)
        indices = np.random.permutation(range(len(X)))[:n_tests]
        test_X = X[indices].to(self.device)
        test_t = t[indices].numpy()
        iterator = DataLoader(
            test_X, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='ensemble/test', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(batch))
        ys = [np.concatenate(_ys) for _ys in ys.values()]
        y = np.array(ys).mean(axis=0)
        metrics = classification_metrics(y, test_t)
        self.threshold = metrics['threshold']

    def predict_proba(self, X):
        pred_X = X.to(self.device)
        iterator = DataLoader(
            pred_X, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='submit', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(batch))
        ys = np.concatenate(
            [np.concatenate(_ys) for _ys in ys.values()], axis=1)
        y = ys.mean(axis=1, keepdims=True)
        return y
