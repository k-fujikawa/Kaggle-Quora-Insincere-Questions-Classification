from qiqc.modules.wrappers.base import NNModuleWrapperBase
from qiqc.registry import AGGREGATOR_REGISTRY


class AggregatorWrapper(NNModuleWrapperBase):

    default_config = None
    registry = AGGREGATOR_REGISTRY

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = self.registry[config.aggregator]()

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--aggregator',
                            choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X, mask):
        h = self.module(X, mask)
        return h
