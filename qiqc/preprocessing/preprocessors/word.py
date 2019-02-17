import numpy as np

from qiqc.preprocessing.modules.vocab import WordVocab
from qiqc.utils import pad_sequence
from qiqc.utils import ApplyNdArray, Pipeline


class WordbasedPreprocessor():

    def tokenize(self, datasets, normalizer, tokenizer):
        tokenize = Pipeline(normalizer, tokenizer)
        apply_tokenize = ApplyNdArray(tokenize, processes=2, dtype=object)
        tokens = [apply_tokenize(d.df.question_text.values) for d in datasets]
        return tokens

    def build_vocab(self, datasets, config):
        train_dataset, test_dataset, submit_dataset = datasets
        vocab = WordVocab(mincount=config.vocab_mincount)
        vocab.add_documents(train_dataset.positives.tokens, 'train-pos')
        vocab.add_documents(train_dataset.negatives.tokens, 'train-neg')
        vocab.add_documents(test_dataset.positives.tokens, 'test-pos')
        vocab.add_documents(test_dataset.negatives.tokens, 'test-neg')
        vocab.add_documents(submit_dataset.df.tokens, 'submit')
        vocab.build()
        return vocab

    def build_tokenids(self, datasets, vocab, config):
        token2id = lambda xs: pad_sequence(  # NOQA
            [vocab.token2id[x] for x in xs], config.maxlen)
        apply_token2id = ApplyNdArray(
            token2id, processes=1, dtype='i', dims=(config.maxlen,))
        tokenids = [apply_token2id(d.df.tokens.values) for d in datasets]
        return tokenids

    def build_sentence_features(self, datasets, sentence_extra_featurizer):
        train_dataset, test_dataset, submit_dataset = datasets
        apply_featurize = ApplyNdArray(
            sentence_extra_featurizer, processes=1, dtype='f',
            dims=(sentence_extra_featurizer.n_dims,))
        _X2 = [apply_featurize(d.df.question_text.values) for d in datasets]
        _train_X2, _test_X2, _submit_X2 = _X2
        train_X2 = sentence_extra_featurizer.fit_standardize(_train_X2)
        test_X2 = sentence_extra_featurizer.standardize(_test_X2)
        submit_X2 = sentence_extra_featurizer.standardize(_submit_X2)
        return train_X2, test_X2, submit_X2

    def build_embedding_matrices(self, datasets, word_embedding_featurizer,
                                 vocab, pretrained_vectors):
        pretrained_vectors_merged = np.stack(
            [wv.vectors for wv in pretrained_vectors.values()]).mean(axis=0)
        vocab.unk = (pretrained_vectors_merged == 0).all(axis=1)
        vocab.known = ~vocab.unk
        embedding_matrices = word_embedding_featurizer(
            pretrained_vectors_merged, datasets)
        return embedding_matrices

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))
        embedding = embedding.mean(axis=0)
        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features
