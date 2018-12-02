import contextlib

import numpy as np
import gensim
from gensim.models import Word2Vec, Doc2Vec, FastText


class BaseWordEmbeddingsModelEx(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_pretrained_vector = False

    @contextlib.contextmanager
    def freeze_pretrained_vector(self):
        self._freeze_pretrained_vector = True
        yield
        self._freeze_pretrained_vector = False

    def build_vocab_with_pretraining(self, sentences, pretrained_vector,
                                     **kwargs):
        super().build_vocab(
            sentences=sentences, **kwargs)
        self.initialize_pretrained_vector(pretrained_vector)

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
        self.reset_pretrained_vector()

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
