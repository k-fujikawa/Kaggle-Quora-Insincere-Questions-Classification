import nltk
import torch
from unidecode import unidecode

from qiqc.features import SentenceFeatureTransformer
from qiqc.features import FrequencyBasedSentenceFeatureTransformer
from qiqc.models.aggregator.pooling import AvgPoolingAggregator
from qiqc.models.aggregator.pooling import MaxPoolingAggregator
from qiqc.models.aggregator.pooling import SumPoolingAggregator
from qiqc.models.aggregator.state import BiRNNLastStateAggregator
from qiqc.models.encoder.attention import StandAloneLinearAttention
from qiqc.models.encoder.rnn import LSTMEncoder
from qiqc.models.encoder.rnn import GRUEncoder
from qiqc.models.encoder.rnn import LSTMGRUEncoder
from qiqc.models.encoder.rnn import GRULSTMEncoder
from qiqc.models.ensembler.simple import AverageEnsembler
from qiqc.models.ensembler.stacking import LinearEnsembler
from qiqc.models.ensembler.stacking import MLPEnsembler
from qiqc.models.ensembler.stacking import LGBMEnsembler

from qiqc.utils import Pipeline
from qiqc.preprocessors import cylower
from qiqc.preprocessors import cysplit
from qiqc.preprocessors import PunctSpacer
from qiqc.preprocessors import NumberReplacer
from qiqc.preprocessors import MisspellReplacer
from qiqc.preprocessors import KerasFilterReplacer


aggregators = {
    'max': MaxPoolingAggregator(),
    'avg': AvgPoolingAggregator(),
    'sum': SumPoolingAggregator(),
    'last': BiRNNLastStateAggregator(),
}
preprocessors = {
    'lower': cylower,
    'punct': PunctSpacer(),
    'punct_edge': PunctSpacer(edge_only=True),
    'unidecode': unidecode,
    'number': NumberReplacer(),
    'number+underscore': NumberReplacer(with_underscore=True),
    'misspell': MisspellReplacer(),
    'keras': KerasFilterReplacer(),
}
tokenizers = {
    'space': cysplit,
    'word_tokenize': nltk.word_tokenize,
}
sentence_features = {
    'frequency': FrequencyBasedSentenceFeatureTransformer(),
}
encoders = {
    'lstm': LSTMEncoder,
    'gru': GRUEncoder,
    'lstmgru': LSTMGRUEncoder,
    'grulstm': GRULSTMEncoder,
}
ensemblers = {
    'avg': AverageEnsembler,
    'linear': LinearEnsembler,
    'mlp': MLPEnsembler,
    'lgbm': LGBMEnsembler,
}
optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
schedulers = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}
attentions = {
    'standalone': StandAloneLinearAttention
}


def build_attention(name):
    return attentions[name]


def build_aggregator(name):
    return aggregators[name]


def build_preprocessor(names):
    assert isinstance(names, list)
    return Pipeline(*[
        preprocessors[n] for n in names
    ])


def build_tokenizer(name):
    return tokenizers[name]


def build_sent2vec(names):
    assert isinstance(names, (list, type(None)))
    if names is None:
        return SentenceFeatureTransformer()
    else:
        return SentenceFeatureTransformer(*[
            sentence_features[n] for n in names
        ])


def build_encoder(name):
    return encoders[name]


def build_ensembler(name):
    return ensemblers[name]


def build_optimizer(config, model):
    optimizer_cls = optimizers[config['name']]
    optimizer = optimizer_cls(model.parameters(), **config['params'])
    return optimizer
