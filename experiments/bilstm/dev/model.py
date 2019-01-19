import torch
import numpy as np
from torch import nn

from gensim.models import Word2Vec

from qiqc.builder import build_attention
from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.features import WordFeature
from qiqc.models import BinaryClassifier


def build_sampler(batchsize, i_cv, epoch, weights):
    return None


def build_models(config, word_freq, token2id, pretrained_vectors, df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    external_vectors = np.stack(
        [wv.vectors for wv in pretrained_vectors.values()])
    external_vectors = external_vectors.mean(axis=0)
    word_features = WordFeature(
        word_freq, token2id, external_vectors, config['vocab']['min_count'])

    if config['model']['embed']['finetune']:
        word_features.finetune(Word2Vec, df)

    if config['model']['embed']['extra_features'] is not None:
        word_features.prepare_extra_features(
            df, token2id, config['model']['embed']['extra_features'])

    for i in range(config['cv']):
        add_noise = config['model']['embed']['add_noise']
        embedding_vectors = word_features.build_feature(add_noise=add_noise)
        embedding = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_vectors), freeze=True)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)

    return models, word_features.unk


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


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
        config['encoder']['n_input'] = self.embedding.embedding_dim
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
