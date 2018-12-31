from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qiqc.model_selection import classification_metrics


class BaseEnsembler(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, t, test_size=0.1):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X):
        y = self.predict_proba(X)
        return (y > self.threshold).astype('i')


class BaseStackingEnsembler(BaseEnsembler, nn.Module):

    def __init__(self, config, models, results, predictor, lossfunc):
        super().__init__()
        self.config = config
        self.models = models
        self.results = results
        self.device = config['device']
        self.batchsize_train = config['batchsize']
        self.batchsize_valid = config['batchsize_valid']
        self.epochs = config['ensembler']['epochs']
        self.predictor = predictor
        self.lossfunc = lossfunc

    @abstractmethod
    def predict_features(self, X):
        pass

    def fit(self, X, t, test_size=0.1):
        # Build dataset with predicted values
        X = X.to(self.device)
        X_iter = DataLoader(X, batch_size=self.batchsize_valid, shuffle=False)
        ys = []
        for batch in tqdm(X_iter, desc='ensemble/preprocess', leave=False):
            ys.append(self.predict_features(batch))
        X = torch.Tensor(np.concatenate(ys))

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

        for epoch in range(self.epochs):
            for batch in tqdm(train_iter, desc='ensemble/train', leave=False):
                batch_X, batch_t = batch
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
        ys = []
        for batch in tqdm(iterator, desc='submit', leave=False):
            self.predictor.eval()
            _X = self.predict_features(batch)
            _X = torch.Tensor(_X).to(self.device)
            y = self.predictor(_X)
            y = torch.sigmoid(y).cpu().detach().numpy()
            ys.append(y)
        y = np.concatenate(ys)
        return y
