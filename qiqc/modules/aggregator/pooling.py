import numpy as np
import torch
from torch import nn

from qiqc.registry import register_aggregator


@register_aggregator('max')
class MaxPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), -np.inf)
        h = hs.max(dim=1)[0]
        return h


@register_aggregator('sum')
class SumPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), 0)
        h = hs.sum(dim=1)
        return h


@register_aggregator('avg')
class AvgPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), 0)
        h = hs.sum(dim=1)
        maxlen = mask.sum(dim=1)
        h /= maxlen[:, None].type(torch.float)
        return h
