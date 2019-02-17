import torch

from qiqc.registry import register_aggregator


@register_aggregator('last')
class BiRNNLastStateAggregator(object):

    def __call__(self, hs, mask):
        batchsize, maxlen, n_hidden = hs.shape
        idx = mask.sum(dim=1) - 1
        zeros = idx * 0
        fw_h = hs[range(batchsize), idx][:, :n_hidden // 2]
        bw_h = hs[range(batchsize), zeros][:, n_hidden // 2:]
        h = torch.cat([fw_h, bw_h], dim=1)
        return h
