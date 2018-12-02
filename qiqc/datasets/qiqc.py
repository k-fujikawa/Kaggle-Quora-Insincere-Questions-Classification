import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class QIQCDataset(Dataset):

    def __init__(self, df, maxlen=100):
        self.df = df
        self.texts = self.df.question_text
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if i == len(self):
            raise StopIteration
        row = self.df.iloc[i]
        return {
            'token_ids': row.get('token_ids'),
            'mask': row.get('token_ids') > 0,
            'tokens': row.get('tokens'),
            'target': row.get('target'),
        }

    def train_test_split(self, test_size, random_state):
        train_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        return QIQCDataset(train_df), QIQCDataset(test_df)


class QIQCTrainDataset(QIQCDataset):

    def __init__(self, datapath='/src/input/train.csv', nrows=None):
        df = pd.read_csv(datapath, nrows=nrows)
        n_labels = {
            0: (df.target == 0).sum(),
            1: (df.target == 1).sum(),
        }
        df['weights'] = df.target.apply(lambda t: 1 / n_labels[t])
        super().__init__(df)


class QIQCSubmitDataset(QIQCDataset):

    def __init__(self, datapath='/src/input/test.csv', nrows=None):
        df = pd.read_csv(datapath, nrows=nrows)
        super().__init__(df)
