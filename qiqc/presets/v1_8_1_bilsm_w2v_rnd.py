import numpy as np
from torch import nn
from qiqc.config import ExperimentConfigBuilderBase
from qiqc.preprocessing.modules import TextNormalizerWrapper
from qiqc.preprocessing.modules import TextTokenizerWrapper
from qiqc.preprocessing.modules import WordEmbeddingFeaturizerWrapper
from qiqc.preprocessing.modules import WordExtraFeaturizerWrapper
from qiqc.preprocessing.modules import SentenceExtraFeaturizerWrapper
from qiqc.preprocessing.preprocessors import WordbasedPreprocessor
from qiqc.modules import EmbeddingWrapper
from qiqc.modules import EncoderWrapper
from qiqc.modules import AggregatorWrapper
from qiqc.modules import MLPWrapper
from qiqc.modules import AverageEnsembler


# =======  Experiment configuration  =======

class ExperimentConfigBuilderPresets(ExperimentConfigBuilderBase):

    default_config = dict(
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=2,
    )


# =======  Preprocessing modules  =======

class TextNormalizerPresets(TextNormalizerWrapper):

    default_config = dict(
        normalizers=[
            'lower',
            'misspell',
            'punct',
            'number+underscore'
        ]
    )


class TextTokenizerPresets(TextTokenizerWrapper):

    default_config = dict(
        tokenizer='space'
    )


class WordEmbeddingFeaturizerPresets(WordEmbeddingFeaturizerWrapper):

    default_config = dict(
        use_pretrained_vectors=['glove', 'paragram'],
        word_embedding_features=['pretrained', 'word2vec'],
    )
    default_extra_config = dict(
        finetune_word2vec_init_unk='zeros',
        finetune_word2vec_mincount=1,
        finetune_word2vec_workers=1,
        finetune_word2vec_iter=5,
        finetune_word2vec_size=300,
        finetune_word2vec_sg=0,
        finetune_word2vec_sorted_vocab=0,
    )


class WordExtraFeaturizerPresets(WordExtraFeaturizerWrapper):

    default_config = dict(
        word_extra_features=[],
    )


class SentenceExtraFeaturizerPresets(SentenceExtraFeaturizerWrapper):

    default_config = dict(
        sentence_extra_features=[],
    )


class PreprocessorPresets(WordbasedPreprocessor):

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))

        # Add noise
        unk = (embedding[0] == 0).all(axis=1)
        mean, std = embedding[0, ~unk].mean(), embedding[0, ~unk].std()
        unk_and_hfq = unk & word_embedding_featurizer.vocab.hfq
        noise = np.random.normal(
            mean, std, (unk_and_hfq.sum(), embedding[0].shape[1]))
        embedding[0, unk_and_hfq] = noise
        embedding[0, 0] = 0

        embedding = embedding.mean(axis=0)
        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features


# =======  Training modules  =======

class EmbeddingPresets(EmbeddingWrapper):

    default_config = dict(
        embedding_dropout1d=0.2,
    )


class EncoderPresets(EncoderWrapper):

    default_config = dict(
        encoder='lstm',
    )
    default_extra_config = dict(
        encoder_bidirectional=True,
        encoder_dropout=0.,
        encoder_n_layers=2,
        encoder_n_hidden=128,
    )


class AggregatorPresets(AggregatorWrapper):

    default_config = dict(
        aggregator='max',
    )


class MLPPresets(MLPWrapper):

    default_config = dict(
        mlp_n_hiddens=[128, 128],
        mlp_bn0=False,
        mlp_dropout0=0.,
        mlp_bn=True,
        mlp_actfun=nn.ReLU(True),
    )


class EnsemblerPresets(AverageEnsembler):
    pass
