import numpy as np
import torch
from torch import nn


class MaxPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        _mask = mask[:, :, None].repeat(1, 1, hs.shape[-1])
        fill_value = torch.full(hs.shape, -np.inf).to(hs.device)
        hs = torch.where(_mask, hs, fill_value)
        h = hs.max(dim=1)[0]
        return h


class SumPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        _mask = mask[:, :, None].repeat(1, 1, hs.shape[-1])
        fill_value = torch.zeros(hs.shape).to(hs.device)
        hs = torch.where(_mask, hs, fill_value)
        h = hs.sum(dim=1)
        return h


class AvgPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        _mask = mask[:, :, None].repeat(1, 1, hs.shape[-1])
        fill_value = torch.zeros(hs.shape).to(hs.device)
        hs = torch.where(_mask, hs, fill_value)
        h = hs.sum(dim=1)
        maxlen = mask.sum(dim=1)
        h /= maxlen[:, None].type(torch.float)
        return h
