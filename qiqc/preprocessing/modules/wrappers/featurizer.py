import numpy as np

from qiqc.registry import SENTENCE_EXTRA_FEATURIZER_REGISTRY
from qiqc.registry import WORD_EMBEDDING_FEATURIZER_REGISTRY
from qiqc.registry import WORD_EXTRA_FEATURIZER_REGISTRY


class WordEmbeddingFeaturizerWrapper(object):

    registry = WORD_EMBEDDING_FEATURIZER_REGISTRY
    default_config = None
    default_extra_config = None

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        self.featurizers = {
            k: self.registry[k](config, vocab)
            for k in config.word_embedding_features}

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--use-pretrained-vectors', nargs='+',
            choices=['glove', 'paragram', 'wnews', 'gnews'])
        parser.add_argument(
            '--word-embedding-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        assert isinstance(cls.default_extra_config, dict)
        for featurizer in config.word_embedding_features:
            cls.registry[featurizer].add_args(parser)
        parser.set_defaults(**cls.default_extra_config)

    def __call__(self, features, datasets):
        return {k: feat(features, datasets)
                for k, feat in self.featurizers.items()}


class WordExtraFeaturizerWrapper(object):

    registry = WORD_EXTRA_FEATURIZER_REGISTRY
    default_config = None

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        self.featurizers = {
            k: self.registry[k]() for k in config.word_extra_features}

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--word-extra-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, vocab):
        empty = np.empty([len(vocab), 0])
        return np.concatenate([empty, *[
            f(vocab) for f in self.featurizers.values()]], axis=1)


class SentenceExtraFeaturizerWrapper(object):

    registry = SENTENCE_EXTRA_FEATURIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.config = config
        self.featurizers = {
            k: self.registry[k]() for k in config.sentence_extra_features}
        self.n_dims = sum(list(f.n_dims for f in self.featurizers.values()))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--sentence-extra-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, sentence):
        empty = np.empty((0,))
        return np.concatenate([empty, *[
            f(sentence) for f in self.featurizers.values()]], axis=0)

    def fit_standardize(self, features):
        assert features.ndim == 2
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std = np.where(self.std != 0, self.std, 1)
        return (features - self.mean) / self.std

    def standardize(self, features):
        assert hasattr(self, 'mean'), hasattr(self, 'std')
        return (features - self.mean) / self.std
