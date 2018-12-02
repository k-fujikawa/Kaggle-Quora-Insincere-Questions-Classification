import csv
import pandas as pd
from gensim.models import KeyedVectors


def load_pretrained_vector(name, test=False):
    if test:
        limit = 1000
    else:
        limit = None

    loader = dict(
        gnews=GNewsPretrainedVector,
        wnews=WNewsPretrainedVector,
        paragram=ParagramPretrainedVector,
        glove=GlovePretrainedVector,
    )

    if name not in loader:
        raise ValueError

    return loader[name].load(limit=limit)


class GNewsPretrainedVector(object):

    name = 'GoogleNews-vectors-negative300'
    path = f'/src/input/embeddings/{name}/{name}.bin'

    @classmethod
    def load(cls, limit=None):
        return KeyedVectors.load_word2vec_format(
            cls.path, binary=True, limit=limit)


class WNewsPretrainedVector(object):

    name = 'wiki-news-300d-1M'
    path = f'/src/input/embeddings/{name}/{name}.vec'

    @classmethod
    def load(cls, limit=None):
        return KeyedVectors.load_word2vec_format(
            cls.path, binary=True, limit=limit)


class ParagramPretrainedVector(object):

    name = 'paragram_300_sl999'
    path = f'/src/input/embeddings/{name}/{name}.txt'

    @classmethod
    def load(cls, limit=None):
        df = pd.read_table(
            cls.path, sep=" ", index_col=0, header=None,
            quoting=csv.QUOTE_NONE, nrows=limit)
        vec = KeyedVectors(300)
        vec.add(df.index, df.values)

        return vec


class GlovePretrainedVector(object):

    name = 'glove.840B.300d'
    path = f'/src/input/embeddings/{name}/{name}.txt'

    @classmethod
    def load(cls, limit=None):
        df = pd.read_table(
            cls.path, sep=" ", index_col=0, header=None,
            quoting=csv.QUOTE_NONE, nrows=limit)
        vec = KeyedVectors(300)
        vec.add(df.index, df.values)

        return vec
