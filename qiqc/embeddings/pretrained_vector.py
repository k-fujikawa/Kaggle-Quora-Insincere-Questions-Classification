from functools import partial
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors


def load_pretrained_vectors(names, tokens, test=False):
    assert isinstance(names, list)
    with Pool(processes=len(names)) as pool:
        f = partial(load_pretrained_vector, tokens=tokens, test=test)
        vectors = pool.map(f, names)
    return dict([(n, v) for n, v in zip(names, vectors)])


def load_pretrained_vector(name, tokens, test=False):
    assert isinstance(name, str)
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

    return loader[name].load(tokens, limit=limit)


class BasePretrainedVector(object):

    @classmethod
    def load(cls, tokens, limit=None):
        coefs = []
        for i, o in enumerate(
                open(cls.path, encoding="utf8", errors='ignore')):
            token, *vector = o.split(' ')
            if limit is not None and i > limit:
                break
            if len(o) <= 100 or str.lower(token) not in tokens:
                continue
            coefs.append((token, np.array(vector, 'f')))

        embeddings_index = dict(coefs)
        vec = KeyedVectors(300)
        vec.add(list(embeddings_index.keys()), list(embeddings_index.values()))

        return vec


class GNewsPretrainedVector(object):

    name = 'GoogleNews-vectors-negative300'
    path = f'/src/input/embeddings/{name}/{name}.bin'

    @classmethod
    def load(cls, tokens, limit=None):
        return KeyedVectors.load_word2vec_format(
            cls.path, binary=True, limit=limit)


class WNewsPretrainedVector(object):

    name = 'wiki-news-300d-1M'
    path = f'/src/input/embeddings/{name}/{name}.vec'

    @classmethod
    def load(cls, tokens, limit=None):
        return KeyedVectors.load_word2vec_format(
            cls.path, binary=False, limit=limit)


class ParagramPretrainedVector(BasePretrainedVector):

    name = 'paragram_300_sl999'
    path = f'/src/input/embeddings/{name}/{name}.txt'


class GlovePretrainedVector(BasePretrainedVector):

    name = 'glove.840B.300d'
    path = f'/src/input/embeddings/{name}/{name}.txt'
