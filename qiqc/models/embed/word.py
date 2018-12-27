import math

import torch
from torch import nn


class WordEmbedding(nn.Module):

    def __init__(self, n_vocab, n_embed, n_hidden=0, padding_idx=0,
                 hidden_bn=False, freeze_embed=True, pretrained_vectors=None,
                 position=False, dropout=0.):
        super().__init__()
        self.position = position
        self.linear = None
        self.hidden_bn = hidden_bn
        self.word_embed = nn.Embedding(
            n_vocab, n_embed, padding_idx=padding_idx)
        self.out_dim = n_embed
        self.dropout = nn.Dropout(dropout)
        if pretrained_vectors is not None:
            self.word_embed.weight = nn.Parameter(
                torch.Tensor(pretrained_vectors))
            self.word_embed.weight.requires_grad = not freeze_embed
        if n_hidden > 0:
            self.out_dim = n_hidden
            self.linear = nn.Linear(n_embed, n_hidden)
            if hidden_bn:
                raise NotImplementedError
                self.bn = nn.BatchNorm1d(n_hidden)
        if position:
            self.positional_embed = PositionalEmbedding(self.out_dim)

    def forward(self, x):
        h = self.word_embed(x)
        h = self.dropout(h)
        if self.linear is not None:
            h = self.linear(h)
        if self.position:
            h += self.positional_embed(x)
        return h


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
