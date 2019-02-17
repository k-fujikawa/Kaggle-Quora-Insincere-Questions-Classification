from qiqc.registry import NORMALIZER_REGISTRY


class TextNormalizerWrapper(object):

    registry = NORMALIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.normalizers = [self.registry[n] for n in config.normalizers]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--normalizers', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, x):
        for normalizer in self.normalizers:
            x = normalizer(x)
        return x
