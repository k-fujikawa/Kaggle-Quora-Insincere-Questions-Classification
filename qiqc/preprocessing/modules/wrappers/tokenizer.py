from qiqc.registry import TOKENIZER_REGISTRY


class TextTokenizerWrapper(object):

    registry = TOKENIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.tokenizer = self.registry[config.tokenizer]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--tokenizer', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, x):
        return self.tokenizer(x)
