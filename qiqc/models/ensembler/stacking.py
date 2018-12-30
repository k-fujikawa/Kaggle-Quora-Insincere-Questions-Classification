import numpy as np
from torch import nn

from qiqc.models.ensembler.base import BaseStackingEnsembler


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
