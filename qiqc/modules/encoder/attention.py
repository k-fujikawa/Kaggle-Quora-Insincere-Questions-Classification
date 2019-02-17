import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandAloneLinearAttention(nn.Module):

    def __init__(self, n_input):
        super().__init__()
        self.linear = nn.Linear(n_input, 1)

    def forward(self, h, mask):
        batchsize, maxlen, n_input = h.shape
        h_flatten = h.contiguous().view(-1, n_input)
        scores = self.linear(h_flatten)
        fill_value = torch.full(scores.shape, -np.inf).to(h.device)
        scores = torch.where(mask.contiguous().view(-1, 1), scores, fill_value)
        p_attn = F.softmax(scores.view(batchsize, maxlen, 1), dim=1)
        h = h * p_attn

        return h * p_attn


class PairwiseDotAttention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, attn_heads, in_size, out_size, dropout=0.):
        super().__init__()
        assert out_size % attn_heads == 0

        # We assume d_v always equals d_k
        self.out_size_child = out_size // attn_heads
        self.attn_heads = attn_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for _ in range(3)])
        self.output_linear = nn.Linear(out_size, out_size)
        self.attention = PairwiseDotAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if len(mask.shape) == 2:
            batchsize, maxlen = mask.shape
            mask = mask.unsqueeze(1).repeat(1, maxlen, 1).unsqueeze(1)
        elif len(mask.shape) == 4:
            batchsize, _, maxlen, maxlen = mask.shape
        else:
            raise ValueError

        # 1) Do all the linear projections in batch from out_size => h x d_k
        query, key, value = [
            l(x).view(batchsize, maxlen, self.attn_heads, self.out_size_child)
                .transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batchsize, -1, self.attn_heads * self.out_size_child)

        return self.output_linear(x)


class MultiHeadSelfAttention(MultiHeadAttention):

    def forward(self, h, mask):
        return super().forward(h, h, h, mask=mask)
