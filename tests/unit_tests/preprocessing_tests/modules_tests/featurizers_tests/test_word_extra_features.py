from unittest import TestCase

import numpy as np

from qiqc.preprocessing.modules import WordVocab
from qiqc.preprocessing.modules import WordExtraFeaturizerWrapper


class TestIDFWordFeaturizer(TestCase):

    def test_call(self):

        class Config(object):
            word_extra_features = ['idf']

        config = Config()
        vocab = WordVocab()
        vocab.add_documents([
            list('abcd'),
            list('abc'),
            list('ab'),
            list('a'),
        ], 'test')
        vocab.build()

        featurizer = WordExtraFeaturizerWrapper(config, vocab)
        features = featurizer(vocab)

        self.assertEqual(features[0], 0)
        self.assertEqual(features[1], 0)
        self.assertTrue(features[2] < features[3])
        self.assertTrue(features[3] < features[4])


class TestUNKWordFeaturizer(TestCase):

    def test_call(self):

        class Config(object):
            word_extra_features = ['unk']

        config = Config()
        vocab = WordVocab()
        vocab.add_documents([
            list('abcd'),
            list('abc'),
            list('ab'),
            list('a'),
        ], 'test')
        vocab.build()
        vocab.unk = np.random.randint(0, 2, (5))

        featurizer = WordExtraFeaturizerWrapper(config, vocab)
        features = featurizer(vocab)
        self.assertEqual(features[0], 0)
        np.testing.assert_equal(features[1:], vocab.unk[1:][:, None])
