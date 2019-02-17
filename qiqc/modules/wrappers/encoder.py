from torch import nn

from qiqc.registry import ENCODER_REGISTRY


class EncoderWrapper(nn.Module):

    registry = ENCODER_REGISTRY

    def __init__(self, config, in_size):
        super().__init__()
        self.config = config
        self.module = self.registry[config.encoder](config, in_size)
        self.out_size = self.module.out_size

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--encoder', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        assert isinstance(cls.default_extra_config, dict)
        cls.registry[config.encoder].add_args(parser)
        parser.set_defaults(**cls.default_extra_config)

    def forward(self, X, mask):
        h = self.module(X, mask)
        return h
