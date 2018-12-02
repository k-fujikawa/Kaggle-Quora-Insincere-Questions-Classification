import numpy as np
import torch
from torch.utils.data import Dataset


class Word2VecDataset(Dataset):

    def __init__(self, dataset, window):
        self.dataset = dataset
        self.window = window
        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window

    def __len__(self):
        return len(self.dataset)

    def collate(self, batch):
        return {
            'X': torch.cat([b[0] for b in batch]),
            't': torch.cat([b[1] for b in batch]),
        }

    def __getitem__(self, i):
        if i == len(self):
            raise StopIteration
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        context_idx = w + offset
        context_idx = context_idx[
            (context_idx >= 0) * (context_idx < len(self))]
        context = self.dataset[context_idx]
        center = self.dataset[[i] * len(context)]

        return center, context
