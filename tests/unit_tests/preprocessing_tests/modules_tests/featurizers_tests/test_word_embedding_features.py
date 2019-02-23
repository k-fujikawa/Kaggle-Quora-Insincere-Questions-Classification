from unittest import TestCase

import numpy as np
import pandas as pd

from qiqc.preprocessing.modules import WordVocab
from qiqc.preprocessing.modules import WordEmbeddingFeaturizerWrapper


class TestWord2VecFeaturizer(TestCase):

    def setUp(self):
        self.mincount = 3
        self.vocab = WordVocab(self.mincount)
        self.vocab.add_documents([
            list('aa'),
            list('ab'),
            list('abc'),
            list('abcd'),
            list('abcde'),
        ], 'test')
        self.vocab.build()

        unk = np.arange(len(self.vocab)) % 2 == 0
        self.vocab.unk = unk
        self.input_features = np.random.uniform(
            -1, 1, (len(self.vocab), 300)).astype('f')
        self.input_features[unk] = 0

    def test_call(self):

        class Config(object):
            use_pretrained_vectors = ['glove', 'paragram']
            word_embedding_features = ['word2vec']
            finetune_word2vec_init_unk = 'zeros'
            finetune_word2vec_mincount = self.mincount
            finetune_word2vec_workers = 1
            finetune_word2vec_window = 1
            finetune_word2vec_iter = 100
            finetune_word2vec_size = 300
            finetune_word2vec_sg = 0

        df = pd.DataFrame(
            {'tokens': [list('abcd'), list('abcdefg'), list('adefg')]})

        config = Config()
        featurizer = WordEmbeddingFeaturizerWrapper(config, self.vocab)
        output_features = featurizer(self.input_features.copy(), [df])
        is_equal = (self.input_features == output_features['word2vec'])
        is_equal = is_equal.all(axis=1)
        is_zeros = (output_features['word2vec'] == 0).all(axis=1)
        np.testing.assert_equal(is_zeros, self.vocab.lfq)
        np.testing.assert_equal(is_equal, self.vocab.lfq & self.vocab.unk)
