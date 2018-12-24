import nltk

from qiqc.models.aggregator.pooling import AvgPoolingAggregator
from qiqc.models.aggregator.pooling import MaxPoolingAggregator
from qiqc.models.aggregator.pooling import SumPoolingAggregator
from qiqc.models.aggregator.state import BiRNNLastStateAggregator
from qiqc.models.ensembler.simple import AverageEnsembler

from qiqc.preprocessors.pipeline import PreprocessPipeline
from qiqc.preprocessors.normalizer import PunctSpacer
from qiqc.preprocessors.normalizer import NumberReplacer
from qiqc.preprocessors.normalizer import HengZhengMispellReplacer
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
    'hengzheng_mispell': HengZhengMispellReplacer(),
    'keras': KerasFilterReplacer(),
}
tokenizers = {
    'space': str.split,
    'word_tokenize': nltk.word_tokenize,
}
ensemblers = {
    'avg': AverageEnsembler
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


def build_ensembler(name):
    return ensemblers[name]
