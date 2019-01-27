from unittest import TestCase

import numpy as np
import pandas as pd

import qiqc


class TestWordFeatures(TestCase):

    def setUp(self):
        self.vocab = qiqc.features.WordVocab()
        self.vocab.add_documents([
            list('aa'),
            list('ab'),
            list('abc'),
            list('abcd'),
            list('abcde'),
        ], 'test')
        self.vocab.build()
        self.min_count = 3
        unk = np.arange(len(self.vocab)) % 2 == 0
        embed_shape = (len(self.vocab), 300)
        pretrained_vectors = np.random.normal(0, 1, embed_shape)
        pretrained_vectors[unk] = 0
        self.transformer = qiqc.features.WordFeatureTransformer(
            self.vocab, pretrained_vectors, self.min_count)

    def test_finetune_skipgram(self):
        df = pd.DataFrame(
            {'tokens': [list('abcd'), list('abcdefg'), list('adefg')]})
        params = {'size': 300, 'iter': 1, 'min_count': self.min_count}
        fill_unk = 'mean'
        vectors = self.transformer.finetune_skipgram(df, params, fill_unk)
        lfq = self.transformer.lfq
        is_equal = vectors == self.transformer.initialW
        np.testing.assert_equal(is_equal.all(axis=1), lfq)

    def test_finetune_fasttext(self):
        df = pd.DataFrame(
            {'tokens': [list('abcd'), list('abcdefg'), list('adefg')]})
        params = {'size': 300, 'iter': 1, 'min_count': self.min_count,
                  'min_n': 1}
        fill_unk = 'zeros'
        vectors = self.transformer.finetune_fasttext(df, params, fill_unk)
        is_equal = vectors == self.transformer.initialW
        np.testing.assert_equal(is_equal.all(axis=1), False)

    def test_standardize(self):
        unk = self.transformer.unk
        n_vocab, n_embed = len(self.vocab), 300
        embedding = np.random.uniform(size=(n_vocab, n_embed))
        embedding[unk] = 0
        _embedding = self.transformer.standardize(embedding)
        np.testing.assert_equal((_embedding == 0).all(axis=1), unk)
        np.testing.assert_almost_equal(_embedding[~unk].mean(axis=0), 0)
        np.testing.assert_almost_equal(_embedding[~unk].std(axis=0), 1)

    def test_standardize_freq(self):
        unk = self.transformer.unk
        n_vocab, n_embed = len(self.vocab), 300
        embedding = np.random.uniform(size=(n_vocab, n_embed))
        embedding[unk] = 0
        _embedding = self.transformer.standardize_freq(embedding)

        freqs = np.array(list(self.vocab.word_freq.values()))[~unk]
        _embedding_repeat = np.repeat(embedding[~unk], freqs, axis=0)
        _mean = _embedding_repeat.mean(axis=0)
        _std = _embedding_repeat.std(axis=0)

        np.testing.assert_equal((_embedding == 0).all(axis=1), unk)
        np.testing.assert_almost_equal(
            (_embedding[~unk] * _std) + _mean, embedding[~unk])

        self.transformer.vocab.word_freq = {
            k: 1 for k in self.transformer.word_freq}
        _embedding = self.transformer.standardize_freq(embedding)
        np.testing.assert_equal((_embedding == 0).all(axis=1), unk)
        np.testing.assert_almost_equal(_embedding[~unk].mean(axis=0), 0)
        np.testing.assert_almost_equal(_embedding[~unk].std(axis=0), 1)
