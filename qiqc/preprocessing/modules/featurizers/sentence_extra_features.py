import numpy as np

from qiqc.registry import register_sentence_extra_features


@register_sentence_extra_features('char')
class CharacterStatisticsFeaturizer(object):

    n_dims = 3

    def __call__(self, sentence):
        feature = {}
        feature['n_chars'] = len(sentence)
        feature['n_caps'] = sum(1 for char in sentence if char.isupper())
        feature['caps_rate'] = feature['n_caps'] / feature['n_chars']
        features = np.array(list(feature.values()))
        return features


@register_sentence_extra_features('word')
class WordStatisticsFeaturizer(object):

    n_dims = 3

    def __call__(self, sentence):
        feature = {}
        tokens = sentence.split()
        feature['n_words'] = len(tokens)
        feature['unique_words'] = len(set(tokens))
        feature['unique_rate'] = feature['unique_words'] / feature['n_words']
        features = np.array(list(feature.values()))
        return features
