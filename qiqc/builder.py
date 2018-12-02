from qiqc.models.aggregator.pooling import AvgPoolingAggregator
from qiqc.models.aggregator.pooling import MaxPoolingAggregator
from qiqc.models.aggregator.pooling import SumPoolingAggregator
from qiqc.models.aggregator.state import BiRNNLastStateAggregator


aggregators = {
    'max': MaxPoolingAggregator(),
    'avg': AvgPoolingAggregator(),
    'sum': SumPoolingAggregator(),
    'last': BiRNNLastStateAggregator(),
}


def build_aggregator(name):
    if name in aggregators:
        return aggregators[name]
    else:
        raise ValueError
