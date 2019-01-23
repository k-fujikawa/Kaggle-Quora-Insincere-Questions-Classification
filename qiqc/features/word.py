from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from scipy.stats import chi2_contingency

from qiqc.utils import parallel_apply


class WordVocab(object):

    def __init__(self):
        self.counter = Counter()
        self.n_documents = 0
        self._counters = {}
        self._n_documents = defaultdict(int)

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
            **{'<PAD>': 1},
            **counter,
        }
        self.token2id = {
            **{'<PAD>': 0},
            **{word: i + 1 for i, word in enumerate(counter)}
        }


class WordFeatureTransformer(object):

    def __init__(self, vocab, initialW, min_count):
        self.vocab = vocab
        self.word_freq = vocab.word_freq
        self.token2id = vocab.token2id
        self.initialW = initialW
        self.finetuned_vectors = None
        self.min_count = min_count

        self.unk = (initialW == 0).all(axis=1)
        self.known = ~self.unk
        self.lfq = np.array(list(vocab.word_freq.values())) < min_count
        self.hfq = ~self.lfq
        self.mean = initialW[self.known].mean()
        self.std = initialW[self.known].std()
        self.extra_features = None

    def finetune_skipgram(self, df, params):
        tokens = df.tokens.values
        model = Word2Vec(**params)
        model.build_vocab_from_freq(self.word_freq)
        initialW = self.initialW.copy()
        initialW[self.unk] = self.mean
        idxmap = np.array(
            [self.vocab.token2id[w] for w in model.wv.index2entity])
        model.wv.vectors[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        model.train(tokens, total_examples=len(tokens), epochs=model.epochs)
        finetunedW = self.initialW.copy()
        finetunedW[idxmap] = model.wv.vectors
        return finetunedW

    def finetune_fasttext(self, df, params):
        tokens = df.tokens.values
        model = FastText(**params)
        model.build_vocab_from_freq(self.word_freq)
        initialW = self.initialW.copy()
        initialW[self.unk] = self.mean
        idxmap = np.array(
            [self.vocab.token2id[w] for w in model.wv.index2entity])
        model.wv.vectors[:] = initialW[idxmap]
        model.wv.vectors_vocab[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        model.train(tokens, total_examples=len(tokens), epochs=model.epochs)
        finetunedW = np.zeros((initialW.shape), 'f')
        for i, word in enumerate(self.vocab.token2id):
            if word in model.wv:
                finetunedW[i] = model.wv.get_vector(word)
        return finetunedW

    def standardize(self, embedding):
        indices = (embedding != 0).all(axis=1)
        _embedding = embedding[indices]
        mean, std = _embedding.mean(axis=0), _embedding.std(axis=0)
        standardized = embedding.copy()
        standardized[indices] = (embedding[indices] - mean) / std
        return standardized

    def standardize_freq(self, embedding):
        indices = (embedding != 0).all(axis=1)
        _embedding = embedding[indices]
        freqs = np.array(list(self.vocab.word_freq.values()))[indices]
        weighted_embedding = _embedding * freqs[:, None]
        mean = weighted_embedding.sum(axis=0) / freqs.sum()
        se = freqs[:, None] * (_embedding - mean) ** 2
        std = np.sqrt(se.sum(axis=0) / freqs.sum())
        standardized = embedding.copy()
        standardized[indices] = (embedding[indices] - mean) / std
        return standardized

    def build_extra_features(self, df, config):
        extra_features = np.empty((len(self.vocab.token2id), 0))
        if 'chi2' in config:
            chi2_features = self._prepare_chi2(df, self.vocab.token2id)
            extra_features = np.concatenate(
                [extra_features, chi2_features], axis=1)
        return extra_features

    # TODO: Fix to build dictionary for calculation efficiency
    def _prepare_chi2(self, df, token2id, threshold=0.01):
        vocab_pos = self.vocab._counters['pos']
        vocab_neg = self.vocab._counters['neg']
        counts = pd.DataFrame({'tokens': list(token2id.keys())})
        counts['TP'], counts['FP'] = 0, 0

        idxmap = [token2id[k] for k, v in vocab_pos.items()]
        counts.loc[idxmap, 'TP'] = list(vocab_pos.values())
        idxmap = [token2id[k] for k, v in vocab_neg.items()]
        counts.loc[idxmap, 'FP'] = list(vocab_neg.values())

        counts['FN'] = self.vocab._n_documents['pos'] - counts.TP
        counts['TN'] = self.vocab._n_documents['neg'] - counts.FP
        counts['TP/.P'] = counts.TP / (counts.TP + counts.FP)
        class_ratio = self.vocab._n_documents['pos'] / self.vocab.n_documents

        def chi2_func(x):
            if x.TN == 0 or x.TP == 0:
                return np.inf
            else:
                return chi2_contingency(np.array(
                    [[x.TP, x.FP], [x.FN, x.TN]]))[1]

        counts['chi2_p'] = parallel_apply(counts, chi2_func, axis=1)
        counts['feature'] = 0
        is_important = (counts.chi2_p < threshold) & \
            (counts['TP/.P'] > class_ratio) & (counts.TP >= self.min_count)
        counts.loc[is_important, 'feature'] = 1
        return counts.feature[:, None]
