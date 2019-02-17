import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from qiqc.utils import ApplyNdArray
from qiqc.registry import register_word_extra_features


@register_word_extra_features('idf')
class IDFWordFeaturizer(object):

    def __call__(self, vocab):
        dfs = np.array(list(vocab.word_freq.values()))
        dfs[0] = vocab.n_documents
        features = np.log(vocab.n_documents / dfs)
        features = features[:, None]
        return features


@register_word_extra_features('unk')
class UnkWordFeaturizer(object):

    def __call__(self, vocab):
        features = vocab.unk.astype('f')
        features[0] = 0
        features = features[:, None]
        return features


@register_word_extra_features('chi2')
class Chi2WordFeaturizer(object):

    def __call__(self, vocab, threshold=0.01):
        vocab_pos = vocab._counters['train-pos']
        vocab_neg = vocab._counters['train-neg']

        counts = pd.DataFrame({'tokens': list(vocab.token2id.keys())})
        counts['TP'], counts['FP'] = 0, 0

        idxmap = [vocab.token2id[k] for k, v in vocab_pos.items()]
        counts.loc[idxmap, 'TP'] = list(vocab_pos.values())
        idxmap = [vocab.token2id[k] for k, v in vocab_neg.items()]
        counts.loc[idxmap, 'FP'] = list(vocab_neg.values())

        counts['FN'] = vocab._n_documents['train-pos'] - counts.TP
        counts['TN'] = vocab._n_documents['train-neg'] - counts.FP
        counts['TP/.P'] = counts.TP / (counts.TP + counts.FP)
        counts['class_ratio'] = vocab._n_documents['train-pos'] / \
            vocab.n_documents
        counts['df'] = (counts.TP + counts.FP) / vocab.n_documents

        def chi2_func(arr):
            TP, FP, FN, TN = arr
            if TN == 0 or TP == 0:
                return np.inf
            else:
                return chi2_contingency(arr.reshape(2, 2))[1]

        threshold = 0.01
        min_count = 10

        apply_chi2 = ApplyNdArray(chi2_func, processes=1, dtype='f')
        counts['chi2_p'] = apply_chi2(
            counts[['TP', 'FP', 'FN', 'TN']].values)
        counts['chi2_label'] = 0
        is_important = (counts.chi2_p < threshold) & \
            (counts['TP/.P'] > counts.class_ratio) & (counts.TP >= min_count)
        counts.loc[is_important, 'chi2_label'] = 1

        return counts.chi2_label[:, None]
