import numpy as np
import torch
from torch import nn

from qiqc.models.ensembler.base import BaseStackingEnsembler
from qiqc.models.fc.mlp import MLP


class LinearEnsembler(BaseStackingEnsembler):

    def __init__(self, config, models, results):
        predictor = nn.Linear(len(models), 1)
        lossfunc = nn.BCEWithLogitsLoss()
        super().__init__(config, models, results, predictor, lossfunc)

    def predict_features(self, X):
        ys = []
        for i, model in enumerate(self.models):
            model.eval()
            ys.append(model.predict_proba(X))
        return np.concatenate(ys, axis=1)


class MLPEnsembler(BaseStackingEnsembler):

    def __init__(self, config, models, results):
        lossfunc = nn.BCEWithLogitsLoss()
        mlp = MLP(
            n_layers=config['ensembler']['mlp']['n_layers'],
            in_size=sum([m.n_hidden + 1 for m in models]),
            out_size=config['ensembler']['mlp']['n_hidden'],
            actfun=nn.ReLU(True),
            bn=config['ensembler']['mlp']['bn'],
        )
        out = nn.Linear(config['ensembler']['mlp']['n_hidden'], 1)
        predictor = nn.Sequential(mlp, out)
        super().__init__(config, models, results, predictor, lossfunc)

    def predict_features(self, X):
        ys = []
        for i, model in enumerate(self.models):
            model.eval()
            _X = model.predict_features(X)
            _y = model.out(_X)
            _X = torch.cat([_X, _y], dim=1).cpu().detach().numpy()
            ys.append(_X)
        return np.concatenate(ys, axis=1)
