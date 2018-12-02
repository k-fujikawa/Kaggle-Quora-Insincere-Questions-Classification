import time
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import optuna
import torch
from sklearn import metrics
from tqdm import tqdm
from tensorboardX import SummaryWriter


# TODO: refactor
def classification_metrics(ys, ts):
    scores = {}
    ys = np.concatenate(ys)
    ts = np.concatenate(ts)

    if len(np.unique(ts)) > 1:
        scores['ap'] = metrics.average_precision_score(ts, ys)
        scores['rocauc'] = metrics.roc_auc_score(ts, ys)
        best_score = -1
        best_threshold = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
            ys_bin = np.digitize(ys, [threshold])
            prec, rec, fbeta, sup = metrics.precision_recall_fscore_support(
                ts, ys_bin, labels=[0, 1], warn_for=[])
            scores[f'prec#{threshold:.2f}'] = prec[1]
            scores[f'rec#{threshold:.2f}'] = rec[1]
            scores[f'fbeta#{threshold:.2f}'] = fbeta[1]
            if best_score < fbeta[1]:
                best_threshold = threshold
                best_score = fbeta[1]
        scores['threshold'] = best_threshold
        scores['fbeta#best'] = best_score

    return scores


def build_metrics(threshold):
    return [
        'dataset', 'loss', 'ap', 'rocauc', 'threshold', 'fbeta#best',
        f'fbeta#{threshold:.2f}', f'prec#{threshold:.2f}',
        f'rec#{threshold:.2f}',
    ]


class Trainer(object):

    def __init__(self, model, optimizer, featurizer, device, outdir):
        self.model = model
        self.optimizer = optimizer
        self.featurizer = featurizer
        self.device = device
        self.outdir = outdir
        if outdir is not None:
            self.writer = SummaryWriter(outdir)
        self.n_trained = 0
        self.epoch = 0

    def calc_loss(self, batch, name='train'):
        maxlen = (batch['mask'] == 1).any(dim=0).sum()
        batch['token_ids'] = batch['token_ids'][:, :maxlen].to(self.device)
        batch['mask'] = batch['mask'][:, :maxlen].to(self.device)
        batch['target'] = batch['target'].to(self.device)
        loss, y = self.model.calc_loss(batch)
        if self.outdir is not None:
            self.writer.add_scalar(f'{name}/loss', loss, self.n_trained)
        output = dict(
            y=torch.sigmoid(y).cpu().data.numpy(),
            t=batch['target'].cpu().data.numpy(),
            loss=loss.cpu().data.numpy(),
        )
        return loss, output

    def evaluate(self, iterator, name='valid'):
        outputs = []
        n_validated = (self.epoch - 1) * len(iterator.dataset)
        for batch in tqdm(iterator, desc=name, leave=False):
            self.model.eval()
            batchsize = len(batch['target'])
            n_validated += batchsize
            loss, output = self.calc_loss(batch, name=name)
            outputs.append(output)
        return pd.DataFrame(outputs)

    def train(self, config, train_iter, valid_iter=None, test_iter=None,
              evaluate_every=np.nan, trial=None):
        bestscore = 0
        bestepoch = 0
        bestscore_valid = 0
        bestmodel = None
        scores = defaultdict(list)
        start = time.time()
        for epoch in tqdm(range(1, config['epochs'] + 1)):
            self.epoch = epoch
            outputs = []
            for i, batch in enumerate(tqdm(train_iter, desc='train')):
                # Train batch
                self.model.train()
                self.optimizer.zero_grad()
                batchsize = batch['token_ids'].size(0)
                self.n_trained += batchsize

                loss, output = self.calc_loss(batch, name='train')
                loss.backward()
                self.optimizer.step()
                outputs.append(output)

                # Evaluate regularly
                if self.n_trained % evaluate_every < batchsize or \
                        (i + 1) == len(train_iter):
                    outputs = pd.DataFrame(outputs)
                    scores['train'].append(dict(
                        dataset='train', n_trained=self.n_trained,
                        loss=outputs.loss.mean(),
                        **classification_metrics(outputs.y, outputs.t)))
                    outputs = self.evaluate(valid_iter, name='valid')
                    scores['valid'].append(dict(
                        dataset='valid', n_trained=self.n_trained,
                        loss=outputs.loss.mean(),
                        **classification_metrics(outputs.y, outputs.t)))
                    fbeta_valid = scores['valid'][-1]['fbeta#best']
                    threshold = scores['valid'][-1]['threshold']

                    if trial is not None:
                        trial.report(-1 * fbeta_valid, self.n_trained)
                        if trial.should_prune(self.n_trained):
                            raise optuna.structs.TrialPruned()
                    if bestscore_valid < scores['valid'][-1]['fbeta#best']:
                        bestepoch = epoch
                        bestscore_valid = scores['valid'][-1]['fbeta#best']
                        outputs = self.evaluate(test_iter, name='test')
                        scores['test'].append(dict(
                            dataset='test', n_trained=self.n_trained,
                            loss=outputs.loss.mean(),
                            **classification_metrics(outputs.y, outputs.t)))
                        bestscore = scores['test'][-1][
                            f'fbeta#{threshold:.2f}']
                        bestmodel = deepcopy(self.model)
                    if self.outdir is not None:
                        valscore = scores['valid'][-1]['threshold']
                        self.writer.add_scalar(
                            'valid/fbeta', valscore, self.n_trained)
                        self.writer.add_scalar(
                            'test/fbeta', bestscore, self.n_trained)
                    outputs = []

                # Print scores only at last batch
                if (i + 1) == len(train_iter):
                    df = pd.DataFrame([v[-1] for v in scores.values()])
                    metrics = build_metrics(threshold)
                    print(f'epoch: {self.epoch}')
                    print(df[metrics])

        return {
            'bestmodel': bestmodel,
            'bestscore': bestscore,
            'bestepoch': bestepoch,
            'time': time.time() - start,
        }
