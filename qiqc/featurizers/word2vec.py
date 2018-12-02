import numpy as np
import sklearn
import torch
from torch.nn.utils.rnn import pad_sequence


class Word2VecFeaturizer(object):

    _reserved = ['<PAD>', '<ABBR>']

    def __init__(self, model, maxlen=100, standardize=True):
        self.model = model
        self.token2id = {'<PAD>': 0, '<UNK>': 1}
        self.vectors = [np.zeros(300, 'f'), np.zeros(300, 'f')]
        self.unk_tokens = {}
        self.maxlen = maxlen
        self.standardize = standardize
        self.scaler = sklearn.preprocessing.StandardScaler()

    def build_w2vtable(self):
        vectors = []
        for word, count in self.model.vocabulary.raw_vocab.items():
            if word in self.model.wv:
                self.token2id[word] = len(self.token2id)
                vectors.append(self.model[word])
            else:
                self.unk_tokens[word] = count
        vectors = np.array(vectors)
        if self.standardize:
            vectors = self.scaler.fit_transform(vectors)
        self.vectors = np.concatenate([self.vectors, vectors])

    def __call__(self, tokens):
        token_ids = []
        for i in range(self.maxlen):
            if i < len(tokens):
                token_id = self.token2id.get(tokens[i], 1)
                token_ids.append(token_id)
            else:
                token_ids.append(0)
        token_ids = torch.Tensor(token_ids).type(torch.long)
        return token_ids

    @property
    def n_vocab(self):
        return len(self.model.wv.vocab)
