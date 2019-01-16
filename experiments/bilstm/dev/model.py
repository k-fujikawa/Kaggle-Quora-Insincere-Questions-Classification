import torch
import numpy as np
from torch import nn

from gensim.models import Word2Vec

from qiqc.builder import build_attention
from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.models import BinaryClassifier


def build_sampler(batchsize, i_cv, epoch, weights):
    return None


def build_models(config, word_freq, token2id, pretrained_vectors, all_df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    external_vectors = np.stack(
        [wv.vectors for wv in pretrained_vectors.values()])
    external_vectors = external_vectors.mean(axis=0)

    unk_indices = (external_vectors == 0).all(axis=1)
    known_indices = ~unk_indices
    lfq_indices = np.array(list(word_freq.values())) < \
        config['vocab']['min_count']
    trainable_unk_indices = ~lfq_indices & unk_indices
    mean = external_vectors[known_indices].mean()
    std = external_vectors[known_indices].std()

    if config['model']['embed']['finetune']:
        wv_finetuned = finetune_embedding(
            Word2Vec, word_freq, external_vectors, all_df)

    for i in range(config['cv']):
        embedding_vectors = external_vectors.copy()
        # Assign noise vectors to unknown high frequency tokens
        if config['model']['embed']['assign_noise']:
            embedding_vectors[trainable_unk_indices] += np.random.normal(
                mean, std, embedding_vectors[trainable_unk_indices].shape)

        # Blend external vectors with local finetuned vectors
        if config['model']['embed']['finetune']:
            embedding_vectors += wv_finetuned.vectors
            embedding_vectors /= 2

        embedding_vectors[lfq_indices & unk_indices] = 0
        embedding = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_vectors), freeze=True)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)

    return models, unk_indices


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


def finetune_embedding(w2vmodel, word_freq, initialW, df):
    n_embed = 300
    tokens = df.tokens.values
    w2v = w2vmodel(
        size=n_embed, min_count=1, workers=1, sorted_vocab=0)
    w2v.build_vocab_from_freq(word_freq)
    w2v.wv.vectors[:] = initialW
    w2v.trainables.syn1neg[:] = initialW
    w2v.train(tokens, total_examples=len(tokens), epochs=5)

    return w2v.wv


class SentenceFeature(object):

    out_size = 6

    def __call__(self, sentence):
        feature = {}
        tokens = sentence.split()
        feature['n_chars'] = len(sentence)
        feature['n_caps'] = sum(1 for char in sentence if char.isupper())
        feature['caps_rate'] = feature['n_caps'] / feature['n_chars']
        feature['n_words'] = len(tokens)
        feature['unique_words'] = len(set(tokens))
        feature['unique_rate'] = feature['unique_words'] / feature['n_words']
        features = np.array(list(feature.values()))
        return features

    def transform(self, train_df, submit_df):
        mean = train_df._X2.values.mean()
        std = train_df._X2.values.std()
        train_df['X2'] = (train_df._X2 - mean) / std
        submit_df['X2'] = (submit_df._X2 - mean) / std


class Encoder(nn.Module):

    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding
        if self.config['embed']['dropout1d'] > 0:
            self.dropout1d = nn.Dropout(config['embed']['dropout1d'])
        if self.config['embed']['dropout2d'] > 0:
            self.dropout2d = nn.Dropout2d(config['embed']['dropout2d'])
        self.encoder = build_encoder(
            config['encoder']['name'])(config['encoder'])
        self.aggregator = build_aggregator(
            config['encoder']['aggregator'])
        if self.config['encoder'].get('attention') is not None:
            self.attn = build_attention(config['encoder']['attention'])(
                config['encoder']['n_hidden'] * config['encoder']['out_scale'])

    def forward(self, X, X2, mask):
        h = self.embedding(X)
        if self.config['embed']['dropout1d'] > 0:
            h = self.dropout1d(h)
        if self.config['embed']['dropout2d'] > 0:
            h = self.dropout2d(h)
        h = self.encoder(h, mask)
        if self.config['encoder'].get('attention') is not None:
            h = self.attn(h, mask)
        h = self.aggregator(h, mask)
        if self.config['encoder']['sentence_features'] > 0:
            h = torch.cat([h, X2], dim=1)
        return h
