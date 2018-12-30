from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qiqc.model_selection import classification_metrics
from qiqc.models.ensembler.base import BaseEnsembler


class LinearEnsembler(BaseEnsembler, nn.Module):

    def __init__(self, config, models, results):
        super().__init__()
        self.config = config
        self.models = models
        self.results = results
        self.device = config['device']
        self.batchsize_train = config['batchsize']
        self.batchsize_valid = config['batchsize_valid']
        self.predictor = nn.Linear(len(models), 1)
        self.lossfunc = nn.BCEWithLogitsLoss()

    def fit(self, X, t, test_size=0.1):
        # Build dataset with predicted values
        X_iter = DataLoader(X, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(X_iter, desc='ensemble/preprocess', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(batch))
        ys = np.concatenate(
            [np.concatenate(_ys) for _ys in ys.values()], axis=1)
        X = torch.Tensor(ys)

        # Ensemble training
        n_tests = int(len(X) * test_size)
        train_indices = np.random.permutation(range(len(X)))[n_tests:]
        test_indices = np.random.permutation(range(len(X)))[:n_tests]

        train_X = X[train_indices].to(self.device)
        train_t = t[train_indices].to(self.device)
        train_dataset = torch.utils.data.TensorDataset(train_X, train_t)
        train_iter = DataLoader(
            train_dataset, batch_size=self.batchsize_valid,
            drop_last=True, shuffle=True)
        model = self.predictor.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for batch in tqdm(train_iter, desc='ensemble/train', leave=False):
            batch_X, batch_t = X
            model.train()
            optimizer.zero_grad()
            loss = self.lossfunc(model(batch_X), batch_t)
            loss.backward()
            optimizer.step()

        test_X = X[test_indices].to(self.device)
        test_t = t[test_indices].numpy()
        test_iter = DataLoader(
            test_X, batch_size=self.batchsize_valid, shuffle=False)

        # Evaluate ensemble results and decide threshold
        ys = []
        for batch in tqdm(test_iter, desc='ensemble/test', leave=False):
            model.eval()
            y = model(batch)
            y = torch.sigmoid(y).cpu().detach().numpy()
            ys.append(y)
        y = np.concatenate(ys)
        metrics = classification_metrics(y, test_t)
        self.threshold = metrics['threshold']
        return y, test_indices, metrics

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
