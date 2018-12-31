import nltk
import torch

from qiqc.models.aggregator.pooling import AvgPoolingAggregator
from qiqc.models.aggregator.pooling import MaxPoolingAggregator
from qiqc.models.aggregator.pooling import SumPoolingAggregator
from qiqc.models.aggregator.state import BiRNNLastStateAggregator
from qiqc.models.encoder.rnn import LSTMEncoder
from qiqc.models.encoder.rnn import LSTMGRUEncoder
from qiqc.models.ensembler.simple import AverageEnsembler
from qiqc.models.ensembler.stacking import LinearEnsembler
from qiqc.models.ensembler.stacking import MLPEnsembler

from qiqc.preprocessors.pipeline import PreprocessPipeline
from qiqc.preprocessors.normalizer import PunctSpacer
from qiqc.preprocessors.normalizer import NumberReplacer
from qiqc.preprocessors.normalizer import MisspellReplacer
from qiqc.preprocessors.normalizer import HengZhengMisspellReplacer
from qiqc.preprocessors.normalizer import NguenMisspellReplacer
from qiqc.preprocessors.normalizer import KerasFilterReplacer


aggregators = {
    'max': MaxPoolingAggregator(),
    'avg': AvgPoolingAggregator(),
    'sum': SumPoolingAggregator(),
    'last': BiRNNLastStateAggregator(),
}
preprocessors = {
    'lower': str.lower,
    'punct': PunctSpacer(),
    'number': NumberReplacer(),
    'number+underscore': NumberReplacer(with_underscore=True),
    'misspell': MisspellReplacer(),
    'hengzheng_misspell': HengZhengMisspellReplacer(),
    'nguen_misspell': NguenMisspellReplacer(),
    'keras': KerasFilterReplacer(),
}
tokenizers = {
    'space': str.split,
    'word_tokenize': nltk.word_tokenize,
}
encoders = {
    'lstm': LSTMEncoder,
    'lstmgru': LSTMGRUEncoder,
}
ensemblers = {
    'avg': AverageEnsembler,
    'linear': LinearEnsembler,
    'mlp': MLPEnsembler,
}
optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
schedulers = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}


def build_aggregator(name):
    return aggregators[name]


def build_preprocessor(names):
    assert isinstance(names, list)
    return PreprocessPipeline(*[
        preprocessors[n] for n in names
    ])


def build_tokenizer(name):
    return tokenizers[name]


def build_encoder(name):
    return encoders[name]


def build_ensembler(name):
    return ensemblers[name]


def build_optimizer(config, model):
    optimizer_cls = optimizers[config['name']]
    optimizer = optimizer_cls(model.parameters(), **config['params'])
    return optimizer
