import numpy as np
import torch


class BiRNNLastStateAggregator(object):

    def __call__(self, hs, mask):
        batchsize, maxlen, n_hidden = hs.shape
        n_hidden_half = n_hidden // 2
        idx = mask.sum(dim=1) - 1
        zeros = idx * 0
        fw_h = hs[range(batchsize), idx][:, :n_hidden // 2]
        bw_h = hs[range(batchsize), zeros][:, n_hidden // 2:]
        h = torch.cat([fw_h, bw_h], dim=1)
        return h
