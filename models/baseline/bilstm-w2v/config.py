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
from qiqc.modules import BinaryClassifier
from qiqc.modules import AverageEnsembler as Ensembler  # NOQA


# =======  Experiment configuration  =======

class ExperimentConfigBuilder(ExperimentConfigBuilderBase):

    default_config = dict(
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[4],
        validate_from=0,
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


# =======  Preprocessing modules  =======

class TextNormalizer(TextNormalizerWrapper):

    default_config = dict(
        normalizers=[
            'lower',
            'misspell',
            'punct',
            'number+underscore'
        ]
    )


class TextTokenizer(TextTokenizerWrapper):

    default_config = dict(
        tokenizer='space'
    )


class WordEmbeddingFeaturizer(WordEmbeddingFeaturizerWrapper):

    default_config = dict(
        use_pretrained_vectors=['glove', 'paragram'],
        word_embedding_features=['pretrained', 'word2vec'],
        word_embedding_purturbation_unk_lfq='noise',
    )
    default_extra_config = dict(
        finetune_word2vec_init_unk='zeros',
        finetune_word2vec_mincount=1,
        finetune_word2vec_workers=1,
        finetune_word2vec_iter=5,
        finetune_word2vec_size=300,
        finetune_word2vec_sg=0,
    )


class WordExtraFeaturizer(WordExtraFeaturizerWrapper):

    default_config = dict(
        word_extra_features=['idf', 'unk'],
    )


class SentenceExtraFeaturizer(SentenceExtraFeaturizerWrapper):

    default_config = dict(
        sentence_extra_features=['char', 'word'],
    )


class Preprocessor(WordbasedPreprocessor):
    pass


# =======  Training modules  =======

def build_model(config, embedding_matrix):
    embedding = Embedding(config, embedding_matrix)
    encoder = Encoder(config, embedding.out_size)
    aggregator = Aggregator(config)
    mlp = MLP(config, encoder.out_size)
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


class Embedding(EmbeddingWrapper):

    default_config = dict(
        embedding_dropout1d=0.2,
    )


class Encoder(EncoderWrapper):

    default_config = dict(
        encoder='lstm',
    )
    default_extra_config = dict(
        encoder_bidirectional=True,
        encoder_dropout=0.,
        encoder_n_layers=2,
        encoder_n_hidden=128,
    )


class Aggregator(AggregatorWrapper):

    default_config = dict(
        aggregator='max',
    )


class MLP(MLPWrapper):

    default_config = dict(
        mlp_n_hiddens=[128, 128],
        mlp_bn0=False,
        mlp_dropout0=0.,
        mlp_bn=True,
        mlp_actfun=nn.ReLU(True),
    )
