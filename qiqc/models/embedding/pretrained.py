from functools import partial
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors


def load_pretrained_vectors(names, token2id, test=False):
    assert isinstance(names, list)
    with Pool(processes=len(names)) as pool:
        f = partial(load_pretrained_vector, token2id=token2id, test=test)
        vectors = pool.map(f, names)
    return dict([(n, v) for n, v in zip(names, vectors)])


def load_pretrained_vector(name, token2id, test=False):
    loader = dict(
        gnews=GNewsPretrainedVector,
        wnews=WNewsPretrainedVector,
        paragram=ParagramPretrainedVector,
        glove=GlovePretrainedVector,
    )
    return loader[name].load(token2id, test)


class BasePretrainedVector(object):

    @classmethod
    def load(cls, token2id, test=False):
        limit = 1000 if test else 1000
        freqs = np.zeros((len(token2id)), dtype='f')
        vectors = np.zeros((len(token2id), 300), dtype='f')
        for i, o in enumerate(
                open(cls.path, encoding="utf8", errors='ignore')):
            token, *vector = o.split(' ')
            token = str.lower(token)
            if limit is not None and i > limit:
                break
            if len(o) <= 100 or token not in token2id:
                continue
            freqs[token2id[token]] += 1
            vectors[token2id[token]] += np.array(vector, 'f')

        vectors[freqs != 0] /= freqs[freqs != 0][:, None]
        vec = KeyedVectors(300)
        vec.add(list(token2id.keys()), vectors)

        return vec


class GNewsPretrainedVector(object):

    name = 'GoogleNews-vectors-negative300'
    path = f'/src/input/embeddings/{name}/{name}.bin'

    @classmethod
    def load(cls, tokens, limit=None):
        raise NotImplementedError
        return KeyedVectors.load_word2vec_format(
            cls.path, binary=True, limit=limit)


class WNewsPretrainedVector(BasePretrainedVector):

    name = 'wiki-news-300d-1M'
    path = f'/src/input/embeddings/{name}/{name}.vec'


class ParagramPretrainedVector(BasePretrainedVector):

    name = 'paragram_300_sl999'
    path = f'/src/input/embeddings/{name}/{name}.txt'


class GlovePretrainedVector(BasePretrainedVector):

    name = 'glove.840B.300d'
    path = f'/src/input/embeddings/{name}/{name}.txt'
