import numpy as np
import pandas as pd
from sklearn import metrics
from tensorboardX import SummaryWriter


def classification_metrics(ys, ts):
    scores = {}
    ys = np.concatenate(ys)
    ts = np.concatenate(ts)

    if len(np.unique(ts)) > 1:
        # Search optimal threshold
        precs, recs, thresholds = metrics.precision_recall_curve(ts, ys)
        thresholds = np.append(thresholds, 1.001)
        idx = (precs != 0) * (recs != 0)
        precs, recs, thresholds = precs[idx], recs[idx], thresholds[idx]
        fbetas = 2 / (1 / precs + 1 / recs)
        best_idx = np.argmax(fbetas)
        threshold = thresholds[best_idx]
        prec = precs[best_idx]
        rec = recs[best_idx]
        fbeta = fbetas[best_idx]

        scores['ap'] = metrics.average_precision_score(ts, ys)
        scores['rocauc'] = metrics.roc_auc_score(ts, ys)
        scores['threshold'] = threshold
        scores['prec'] = prec
        scores['rec'] = rec
        scores['fbeta'] = fbeta

    return scores


class ClassificationResult(object):

    def __init__(self, name, outdir=None, postfix=None, main_metrics='fbeta'):
        self.initialize()
        self.name = name
        self.postfix = postfix
        self.outdir = outdir
        self.summary = None
        self.main_metrics = main_metrics
        self.n_trained = 0
        if outdir is not None:
            self.writer = SummaryWriter(str(outdir))

    def initialize(self):
        self.losses = []
        self.ys = []
        self.ts = []

    def add_record(self, loss, y, t):
        self.losses.append(loss)
        self.ys.append(y)
        self.ts.append(t)
        self.n_trained += len(y)
        if self.outdir is not None:
            self.writer.add_scalar(
                f'{self.name}/loss/{self.postfix}', loss, self.n_trained)

    def calc_score(self, epoch):
        loss = np.array(self.losses).mean()
        score = classification_metrics(self.ys, self.ts)
        summary = dict(name=self.name, loss=loss, **score)
        if len(score) > 0:
            if self.summary is None:
                self.summary = pd.DataFrame([summary], index=[epoch])
                self.summary.index.name = 'epoch'
            else:
                self.summary.loc[epoch] = summary
        self.initialize()

    def get_dict(self):
        loss, fbeta, epoch = 0, 0, 0
        if self.summary is not None:
            row = self.summary.iloc[-1]
            epoch = row.name
            loss = row.loss
            fbeta = row.fbeta
        return {
            'epoch': epoch,
            'loss': loss,
            'fbeta': fbeta,
        }

    @property
    def fbeta(self):
        if self.summary is None:
            return 0
        else:
            return self.summary.fbeta[-1]

    @property
    def best_fbeta(self):
        return self.summary[self.main_metrics].max()

    @property
    def best_epoch(self):
        return self.summary[self.main_metrics].idxmax()

    @property
    def best_threshold(self):
        idx = self.summary[self.main_metrics].idxmax()
        return self.summary['threshold'][idx]
