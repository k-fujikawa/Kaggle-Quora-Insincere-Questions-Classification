from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qiqc.modules.ensembler.base import BaseEnsembler


class AverageEnsembler(BaseEnsembler):

    def __init__(self, config, models, results):
        self.config = config
        self.models = models
        self.results = results
        self.device = config.device
        self.batchsize_train = config.batchsize
        self.batchsize_valid = config.batchsize_valid
        self.threshold_cv = np.array(
            [m.threshold for m in models]).mean()
        self.threshold = self.threshold_cv

    def fit(self, X, X2, t, test_size=0.1):
        # Nothing to do
        pass

    def predict_proba(self, X, X2):
        pred_X = X.to(self.device)
        pred_X2 = X2.to(self.device)
        dataset = torch.utils.data.TensorDataset(pred_X, pred_X2)
        iterator = DataLoader(
            dataset, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='submit', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(*batch))
        ys = np.concatenate(
            [np.concatenate(_ys) for _ys in ys.values()], axis=1)
        y = ys.mean(axis=1, keepdims=True)
        return y
