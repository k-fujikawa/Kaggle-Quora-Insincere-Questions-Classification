from collections import Counter, defaultdict

import numpy as np


class WordVocab(object):

    def __init__(self, mincount=1):
        self.counter = Counter()
        self.n_documents = 0
        self._counters = {}
        self._n_documents = defaultdict(int)
        self.mincount = mincount

    def __len__(self):
        return len(self.token2id)

    def add_documents(self, documents, name):
        self._counters[name] = Counter()
        for document in documents:
            bow = dict.fromkeys(document, 1)
            self._counters[name].update(bow)
            self.counter.update(bow)
            self.n_documents += 1
            self._n_documents[name] += 1

    def build(self):
        counter = dict(self.counter.most_common())
        self.word_freq = {
            **{'<PAD>': 0},
            **counter,
        }
        self.token2id = {
            **{'<PAD>': 0},
            **{word: i + 1 for i, word in enumerate(counter)}
        }
        self.lfq = np.array(list(self.word_freq.values())) < self.mincount
        self.hfq = ~self.lfq
