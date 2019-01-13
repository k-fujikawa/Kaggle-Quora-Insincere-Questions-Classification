import torch
import numpy as np
from torch import nn

from gensim.models import Word2Vec

from qiqc.builder import build_attention
from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.models import BinaryClassifier


def build_models(config, word_freq, token2id, pretrained_vectors, all_df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    external_vectors = np.concatenate(
        [wv.vectors[None, :] for wv in pretrained_vectors.values()])
    external_vectors = external_vectors.mean(axis=0)

    unk_indices = (external_vectors == 0).all(axis=1)
    known_indices = ~unk_indices
    lfq_indices = np.array(list(word_freq.values())) < \
        config['vocab']['min_count']
    trainable_unk_indices = ~lfq_indices & unk_indices
    mean = external_vectors[known_indices].mean()
    std = external_vectors[known_indices].std()

    if config['embedding']['finetune']:
        wv_finetuned = finetune_embedding(
            config, word_freq, token2id, external_vectors, all_df)

    for i in range(config['cv']):
        embedding_vectors = external_vectors.copy()
        # Assign noise vectors to unknown high frequency tokens
        if config['embedding']['assign_noise']:
            embedding_vectors[trainable_unk_indices] += np.random.normal(
                mean, std, embedding_vectors[trainable_unk_indices].shape)

        # Blend external vectors with local finetuned vectors
        if config['embedding']['finetune']:
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


def finetune_embedding(config, word_freq, token2id, initial_vectors, all_df):
    n_embed = 300
    tokens = all_df.tokens.values
    w2v = Word2Vec(
        size=n_embed, min_count=1, workers=1, sorted_vocab=0)
    w2v.build_vocab_from_freq(word_freq)
    w2v.wv.vectors[:] = initial_vectors
    w2v.trainables.syn1neg[:] = initial_vectors
    assert np.allclose(
        initial_vectors[token2id['the']], w2v.wv.get_vector('the'))
    w2v.train(tokens, total_examples=len(tokens), epochs=5)
    assert (w2v.wv.get_vector('<PAD>') == 0).all()

    return w2v.wv


class Encoder(nn.Module):

    def __init__(self, config, embedding):
        super().__init__()
        self.embedding = embedding
        self.dropout1d = nn.Dropout(config['embed']['dropout1d'])
        self.dropout2d = nn.Dropout2d(config['embed']['dropout2d'])
        self.encoder = build_encoder(
            config['encoder']['name'])(config['encoder'])
        self.aggregator = build_aggregator(
            config['encoder']['aggregator'])
        self.attn = None
        if config['encoder'].get('attention') is not None:
            self.attn = build_attention(config['encoder']['attention'])(
                config['encoder']['n_hidden'] * config['encoder']['out_scale'])

    def forward(self, X, mask):
        h = self.embedding(X)
        h = self.dropout1d(h)
        h = self.dropout2d(h)
        h = self.encoder(h, mask)
        if self.attn is not None:
            h = self.attn(h, mask)
        h = self.aggregator(h, mask)
        return h
