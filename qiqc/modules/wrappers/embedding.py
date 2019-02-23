import torch
from torch import nn

from qiqc.modules.wrappers.base import NNModuleWrapperBase


class EmbeddingWrapper(NNModuleWrapperBase):

    default_config = None

    def __init__(self, config, embedding_matrix):
        super().__init__()
        self.config = config
        self.module = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_matrix), freeze=True)
        if self.config.embedding_dropout1d > 0:
            self.dropout1d = nn.Dropout(config.embedding_dropout1d)
        if self.config.embedding_dropout2d > 0:
            self.dropout2d = nn.Dropout2d(config.embedding_dropout2d)
        if self.config.embedding_spatial_dropout > 0:
            self.spatial_dropout = nn.Dropout2d(
                config.embedding_spatial_dropout)
        self.out_size = embedding_matrix.shape[1]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--embedding-dropout1d', type=float, default=0.)
        parser.add_argument('--embedding-dropout2d', type=float, default=0.)
        parser.add_argument('--embedding-spatial-dropout',
                            type=float, default=0.)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X):
        h = self.module(X)
        if self.config.embedding_dropout1d > 0:
            h = self.dropout1d(h)
        if self.config.embedding_dropout2d > 0:
            h = self.dropout2d(h)
        if self.config.embedding_spatial_dropout > 0:
            h = h.permute(0, 2, 1)
            h = self.spatial_dropout(h)
            h = h.permute(0, 2, 1)
        return h
