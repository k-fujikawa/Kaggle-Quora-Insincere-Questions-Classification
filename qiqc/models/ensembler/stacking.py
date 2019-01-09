import lightgbm as lgb
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from qiqc.models.ensembler.base import BaseStackingEnsembler
from qiqc.models.fc.mlp import MLP


class LinearEnsembler(BaseStackingEnsembler):

    def __init__(self, config, models, results):
        super().__init__(config, models, results)
        self.predictor = nn.Linear(len(models), 1)
        self.lossfunc = nn.BCEWithLogitsLoss()

    def predict_features(self, X):
        ys = []
        for i, model in enumerate(self.models):
            model.eval()
            ys.append(model.predict_proba(X))
        return np.concatenate(ys, axis=1)


class MLPEnsembler(BaseStackingEnsembler):

    def __init__(self, config, models, results):
        super().__init__(config, models, results)
        mlp = MLP(
            n_layers=config['ensembler']['params']['n_layers'],
            in_size=sum([m.n_hidden + 1 for m in models]),
            out_size=config['ensembler']['params']['n_hidden'],
            actfun=nn.ReLU(True),
            bn=config['ensembler']['params']['bn'],
        )
        out = nn.Linear(config['ensembler']['params']['n_hidden'], 1)
        self.predictor = nn.Sequential(mlp, out)
        self.lossfunc = nn.BCEWithLogitsLoss()

    def predict_features(self, X):
        ys = []
        for i, model in enumerate(self.models):
            model.eval()
            _X = model.predict_features(X)
            _y = model.out(_X)
            _X = torch.cat([_X, _y], dim=1).cpu().detach().numpy()
            ys.append(_X)
        return np.concatenate(ys, axis=1)


class LGBMEnsembler(BaseStackingEnsembler):

    def fit(self, X, t, test_size=0.1):
        X = self.build_feature(X)
        t = t.numpy().ravel()
        train_X, valid_X, train_t, valid_t = train_test_split(
            X, t, test_size=0.1, random_state=0)
        train_dataset = lgb.Dataset(train_X, label=train_t)
        valid_dataset = lgb.Dataset(
            valid_X, label=valid_t, reference=train_dataset)
        self.predictor = lgb.train(
            self.config['ensembler']['params'],
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            early_stopping_rounds=50,
            verbose_eval=10
        )

    def predict_proba(self, X):
        self.predictor

    def predict_features(self, X):
        ys = []
        for i, model in enumerate(self.models):
            model.eval()
            _X = model.predict_features(X)
            _y = model.out(_X)
            _X = torch.cat([_X, _y], dim=1).cpu().detach().numpy()
            ys.append(_X)
        return np.concatenate(ys, axis=1)
