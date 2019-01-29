import numpy as np


class SentenceFeatureTransformer(object):

    def __init__(self, *transformers):
        self.transformers = transformers
        self.n_dims = sum([t.n_dims for t in self.transformers])

    def __call__(self, sentence):
        return np.concatenate([t(sentence) for t in self.transformers], axis=0)

    def fit_transform(self, features):
        assert features.ndim == 2
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std = np.where(self.std != 0, self.std, 1)
        return (features - self.mean) / self.std

    def transform(self, features):
        assert hasattr(self, 'mean'), hasattr(self, 'std')
        return (features - self.mean) / self.std


class FrequencyBasedSentenceFeatureTransformer(object):

    n_dims = 6

    def __call__(self, sentence):
        feature = {}
        tokens = sentence.split()
        feature['n_chars'] = len(sentence)
        feature['n_caps'] = sum(1 for char in sentence if char.isupper())
        feature['caps_rate'] = feature['n_caps'] / feature['n_chars']
        feature['n_words'] = len(tokens)
        feature['unique_words'] = len(set(tokens))
        feature['unique_rate'] = feature['unique_words'] / feature['n_words']
        features = np.array(list(feature.values()))
        return features
