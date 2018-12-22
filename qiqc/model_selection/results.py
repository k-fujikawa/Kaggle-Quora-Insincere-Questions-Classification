import numpy as np
import pandas as pd
from sklearn import metrics
from tensorboardX import SummaryWriter


# TODO: refactor
def classification_metrics(ys, ts):
    scores = {}
    ys = np.concatenate(ys)
    ts = np.concatenate(ts)

    if len(np.unique(ts)) > 1:
        scores['ap'] = metrics.average_precision_score(ts, ys)
        scores['rocauc'] = metrics.roc_auc_score(ts, ys)
        best_fbeta = -1
        best_threshold = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
            ys_bin = np.digitize(ys, [threshold])
            prec, rec, fbeta, sup = metrics.precision_recall_fscore_support(
                ts, ys_bin, labels=[0, 1], warn_for=[])
            if best_fbeta < fbeta[1]:
                best_threshold = threshold
                best_prec = prec[1]
                best_rec = rec[1]
                best_fbeta = fbeta[1]
        scores['threshold'] = best_threshold
        scores['prec'] = best_prec
        scores['rec'] = best_rec
        scores['fbeta'] = best_fbeta

    return scores


class ClassificationResult(object):

    def __init__(self, name, outdir=None, main_metrics='fbeta'):
        self.initialize()
        self.name = name
        self.outdir = outdir
        self.summary = None
        self.main_metrics = main_metrics
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
