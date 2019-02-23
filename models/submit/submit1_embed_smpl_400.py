import numpy as np
from torch import nn

from qiqc.config import ExperimentConfigBuilderBase
from qiqc.modules import BinaryClassifier
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import TextNormalizerPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import TextTokenizerPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import WordEmbeddingFeaturizerPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import WordExtraFeaturizerPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import SentenceExtraFeaturizerPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import PreprocessorPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EmbeddingPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EncoderPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import AggregatorPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import MLPPresets
from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EnsemblerPresets  # NOQA


# =======  Experiment configuration  =======

class ExperimentConfigBuilder(ExperimentConfigBuilderBase):

    default_config = dict(
        test=False,
        device=None,
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=4,
    )

    @property
    def modules(self):
        return [
            TextNormalizer,
            TextTokenizer,
            WordEmbeddingFeaturizer,
            WordExtraFeaturizer,
            SentenceExtraFeaturizer,
            Embedding,
            Encoder,
            Aggregator,
            MLP,
        ]


def build_model(config, embedding_matrix, n_sentence_extra_features):
    embedding = Embedding(config, embedding_matrix)
    encoder = Encoder(config, embedding.out_size)
    aggregator = Aggregator(config)
    mlp = MLP(config, encoder.out_size + n_sentence_extra_features)
    out = nn.Linear(config.mlp_n_hiddens[-1], 1)
    lossfunc = nn.BCEWithLogitsLoss()

    return BinaryClassifier(
        embedding=embedding,
        encoder=encoder,
        aggregator=aggregator,
        mlp=mlp,
        out=out,
        lossfunc=lossfunc,
    )


# =======  Preprocessing modules  =======

class TextNormalizer(TextNormalizerPresets):
    pass


class TextTokenizer(TextTokenizerPresets):
    pass


class WordEmbeddingFeaturizer(WordEmbeddingFeaturizerPresets):
    pass


class WordExtraFeaturizer(WordExtraFeaturizerPresets):

    default_config = dict(
        word_extra_features=['idf', 'unk'],
    )


class SentenceExtraFeaturizer(SentenceExtraFeaturizerPresets):

    default_config = dict(
        sentence_extra_features=['char', 'word'],
    )


class Preprocessor(PreprocessorPresets):

    embedding_sampling = 400

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))

        # Concat embedding
        embedding = np.concatenate(embedding, axis=1)
        vocab = word_embedding_featurizer.vocab
        embedding[vocab.lfq & vocab.unk] = 0

        # Embedding random sampling
        n_embed = embedding.shape[1]
        n_select = self.embedding_sampling
        idx = np.random.permutation(n_embed)[:n_select]
        embedding = embedding[:, idx]

        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features


# =======  Training modules  =======

class Embedding(EmbeddingPresets):
    pass


class Encoder(EncoderPresets):
    pass


class Aggregator(AggregatorPresets):
    pass


class MLP(MLPPresets):
    pass


class Ensembler(EnsemblerPresets):
    pass
