from copy import deepcopy

import torch
import numpy as np
from torch import nn

from qiqc.builder import build_attention
from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.features import WordFeatureTransformer
from qiqc.models import BinaryClassifier


def build_sampler(batchsize, i_cv, epoch, weights):
    return None


def build_models(config, vocab, pretrained_vectors, df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    external_vectors = np.stack(
        [wv.vectors for wv in pretrained_vectors.values()])
    embeddings = {'external': external_vectors.mean(axis=0)}
    extra = None
    transformer = WordFeatureTransformer(
        vocab, embeddings['external'], config['vocab']['min_count'])

    # Fine-tuning
    if config['model']['embed']['finetune']['skipgram'] is not None:
        embeddings['skipgram'] = transformer.finetune_skipgram(
            df, config['model']['embed']['finetune']['skipgram'])
    if config['model']['embed']['finetune']['fasttext'] is not None:
        embeddings['fasttext'] = transformer.finetune_fasttext(
            df, config['model']['embed']['finetune']['fasttext'])

    # Standardize
    assert config['model']['embed']['standardize'] in {'vocab', 'freq', None}
    if config['model']['embed']['standardize'] == 'vocab':
        embeddings = {k: transformer.standardize(v)
                      for k, v in embeddings.items()}
    elif config['model']['embed']['standardize'] == 'freq':
        embeddings = {k: transformer.standardize_freq(v)
                      for k, v in embeddings.items()}

    # Extra features
    if config['model']['embed']['extra_features'] is not None:
        extra = transformer.prepare_extra_features(
            df, vocab.token2id, config['model']['embed']['extra_features'])

    assert config['model']['embed']['unk_hfq'] in {'noise', None}
    for i in range(config['cv']):
        _embeddings = deepcopy(embeddings)
        if config['model']['embed']['unk_hfq'] == 'noise':
            indices = transformer.unk & transformer.hfq
            _embeddings['external'][indices] += np.random.normal(
                transformer.mean, transformer.std,
                _embeddings['external'][indices].shape)
        embedding_matrix = np.stack(list(_embeddings.values())).mean(axis=0)
        embedding_matrix[transformer.lfq & transformer.unk] = 0

        if config['model']['embed']['extra_features'] is not None:
            embedding_matrix = np.concatenate(
                [embedding_matrix, extra], axis=1)

        embedding = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_matrix), freeze=True)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)

    return models, transformer.unk


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


def build_sentence_feature():
    return None


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
