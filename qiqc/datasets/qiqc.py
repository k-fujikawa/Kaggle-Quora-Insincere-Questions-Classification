import os

import numpy as np
import pandas as pd
import sklearn
import torch


def load_qiqc(n_rows=None):
    train_df = pd.read_csv(f'{os.environ["DATADIR"]}/train.csv', nrows=n_rows)
    submit_df = pd.read_csv(f'{os.environ["DATADIR"]}/test.csv', nrows=n_rows)
    n_labels = {
        0: (train_df.target == 0).sum(),
        1: (train_df.target == 1).sum(),
    }
    train_df['target'] = train_df.target.astype('f')
    train_df['weights'] = train_df.target.apply(lambda t: 1 / n_labels[t])

    return train_df, submit_df


def build_datasets(train_df, submit_df, holdout=False, seed=0):
    submit_dataset = QIQCDataset(submit_df)
    if holdout:
        # Train : Test split for holdout training
        splitter = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed)
        train_indices, test_indices = list(splitter.split(
            train_df, train_df.target))[0]
        train_indices.sort(), test_indices.sort()
        train_dataset = QIQCDataset(
            train_df.iloc[train_indices].reset_index(drop=True))
        test_dataset = QIQCDataset(
            train_df.iloc[test_indices].reset_index(drop=True))
    else:
        train_dataset = QIQCDataset(train_df)
        test_dataset = QIQCDataset(train_df.head(0))

    return train_dataset, test_dataset, submit_dataset


class QIQCDataset(object):

    def __init__(self, df):
        self.df = df

    @property
    def tokens(self):
        return self.df.tokens.values

    @tokens.setter
    def tokens(self, tokens):
        self.df['tokens'] = tokens

    @property
    def positives(self):
        return self.df[self.df.target == 1]

    @property
    def negatives(self):
        return self.df[self.df.target == 0]

    def build(self, device):
        self._X = self.tids
        self.X = torch.Tensor(self._X).type(torch.long).to(device)
        if 'target' in self.df:
            self._t = self.df.target[:, None]
            self._W = self.df.weights
            self.t = torch.Tensor(self._t).type(torch.float).to(device)
            self.W = torch.Tensor(self._W).type(torch.float).to(device)
        if hasattr(self, '_X2'):
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)
        else:
            self._X2 = np.zeros((self._X.shape[0], 1), 'f')
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)

    def build_labeled_dataset(self, indices):
        return torch.utils.data.TensorDataset(
            self.X[indices], self.X2[indices],
            self.t[indices], self.W[indices])
