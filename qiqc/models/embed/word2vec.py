import contextlib

import numpy as np
import gensim
import sklearn
from gensim.models import Word2Vec, Doc2Vec, FastText


class BaseWordEmbeddingsModelEx(object):

    def __init__(self, initialize='zero', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert initialize in {'zero', 'normal'}
        self._freeze_pretrained_vector = False

    @contextlib.contextmanager
    def freeze_pretrained_vector(self):
        self._freeze_pretrained_vector = True
        yield
        self._freeze_pretrained_vector = False

    def initialize_pretrained_vector(self, pretrained_vector):
        self.entities = []
        self.known_freq = {}
        self.unk_freq = {}
        self.weights = []
        for word in self.wv.vocab.keys():
            if word in pretrained_vector.vocab:
                self.known_freq[word] = self.wv.vocab[word].count
                self.entities.append(word)
                self.weights.append(pretrained_vector[word])
            else:
                self.unk_freq[word] = self.wv.vocab[word].count
        self.weights = np.array(self.weights)
        self.initialize_wv(self.weights.mean(), self.weights.std())
        self.reset_pretrained_vector()

    def initialize_wv(self, mean, std):
        self.wv.vectors = np.random.normal(mean, std, self.wv.vectors.shape)

    def reset_pretrained_vector(self, sentences=None):
        if sentences is None:
            entities = self.entities
            weights = self.weights
        else:
            entities = set()
            for sentence in sentences:
                for token in sentence:
                    if token in self.known_freq:
                        entities.add(token)
            entities = list(entities)
            weights = self.wv[entities]
        in_vocab_idxs = [self.wv.vocab[e].index for e in entities]
        self.wv.vectors[in_vocab_idxs] = weights

    def build_embedding_matrix(self, token2id, standardize=False):
        token2vecid = {}
        vectors = [np.zeros(300, 'f')]
        scaler = sklearn.preprocessing.StandardScaler()

        for word, count in token2id.items():
            if word in self.wv:
                token2vecid[word] = len(vectors)
                vectors.append(self[word])
            else:
                token2vecid[word] = 0

        vectors = np.array(vectors)
        if standardize:
            vectors = scaler.fit_transform(vectors)
            vectors[0] = np.zeros(300, 'f')

        embedding_matrix = []
        for token, idx in token2id.items():
            embedding_matrix.append(vectors[token2vecid.get(token, 0)])

        return np.array(embedding_matrix)


class Word2VecEx(BaseWordEmbeddingsModelEx, Word2Vec):

    def _do_train_job(self, sentences, alpha, inits):
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += gensim.models.word2vec.train_batch_sg(
                self, sentences, alpha, work, self.compute_loss)
        else:
            tally += gensim.models.word2vec.train_batch_cbow(
                self, sentences, alpha, work, neu1, self.compute_loss)
        if self._freeze_pretrained_vector:
            self.reset_pretrained_vector(sentences)
        return tally, self._raw_word_count(sentences)


class FastTextEx(BaseWordEmbeddingsModelEx, FastText):
    pass


class Doc2VecEx(BaseWordEmbeddingsModelEx, Doc2Vec):
    pass
