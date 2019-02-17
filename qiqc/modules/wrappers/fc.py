import torch
from torch import nn

from qiqc.modules.wrappers.base import NNModuleWrapperBase


class MLPWrapper(NNModuleWrapperBase):

    default_config = None

    def __init__(self, config, in_size):
        super().__init__()
        self.in_size = in_size
        self.config = config
        assert isinstance(config.mlp_n_hiddens, list)
        layers = []
        if config.mlp_bn0:
            layers.append(nn.BatchNorm1d(in_size))
        if config.mlp_dropout0 > 0:
            layers.append(nn.Dropout(config.mlp_dropout0))
        for n_hidden in config.mlp_n_hiddens:
            layers.append(nn.Linear(in_size, n_hidden))
            if config.mlp_actfun is not None:
                layers.append(config.mlp_actfun)
            if config.mlp_bn:
                layers.append(nn.BatchNorm1d(n_hidden))
            if config.mlp_dropout > 0:
                layers.append(nn.Dropout(config.mlp_dropout))
            in_size = n_hidden
        self.layers = nn.Sequential(*layers)

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--mlp-n-hiddens', type=list)
        parser.add_argument('--mlp-bn', type=bool)
        parser.add_argument('--mlp-bn0', type=bool)
        parser.add_argument('--mlp-dropout', type=float, default=0.)
        parser.add_argument('--mlp-dropout0', type=float, default=0.)
        parser.add_argument('--mlp-actfun', default=0.)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X, X2):
        h = X
        if X.shape[1] + X2.shape[1] == self.in_size:
            h = torch.cat([h, X2], dim=1)
        h = self.layers(h)
        return h
