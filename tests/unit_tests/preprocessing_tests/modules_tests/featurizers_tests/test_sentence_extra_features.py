from unittest import TestCase

import numpy as np
from parameterized import parameterized

from qiqc.preprocessing.modules import SentenceExtraFeaturizerWrapper


class TestCharacterStatisticsFeaturizer(TestCase):

    @parameterized.expand([
        ['A abc b c .'],
        ['abcd'],
        ['ABCD'],
    ])
    def test_call(self, sentence):

        class Config(object):
            sentence_extra_features = ['char']

        config = Config()
        featurizer = SentenceExtraFeaturizerWrapper(config)
        features = featurizer(sentence)

        n_caps = sum(1 for char in sentence if char.isupper())
        np.testing.assert_array_equal(featurizer(sentence).shape, (3,))
        self.assertEqual(features[0], len(sentence))
        self.assertEqual(features[1], n_caps)
        self.assertEqual(features[2], n_caps / len(sentence))


class TestWordStatisticsFeaturizer(TestCase):

    @parameterized.expand([
        ['A abc b c .'],
        ['abcd'],
        ['ABCD ABCD'],
    ])
    def test_call(self, sentence):

        class Config(object):
            sentence_extra_features = ['word']

        config = Config()
        featurizer = SentenceExtraFeaturizerWrapper(config)
        features = featurizer(sentence)

        tokens = sentence.split()
        np.testing.assert_array_equal(featurizer(sentence).shape, (3,))
        self.assertEqual(features[0], len(tokens))
        self.assertEqual(features[1], len(set(tokens)))
        self.assertEqual(features[2], len(set(tokens)) / len(tokens))
