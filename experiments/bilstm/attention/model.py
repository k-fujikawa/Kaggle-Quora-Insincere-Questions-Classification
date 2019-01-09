import torch
import numpy as np
from torch import nn

from gensim.models import Word2Vec

from qiqc.builder import build_attention
from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.embeddings import build_word_vectors
from qiqc.models import BinaryClassifier
from qiqc.models import MultiHeadSelfAttention


def build_models(config, word_freq, token2id, pretrained_vectors, all_df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    embedding, unk_freq = build_embedding(
        config, word_freq, token2id, pretrained_vectors, all_df)
    for i in range(config['cv']):
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)
    return models, unk_freq


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


def build_embedding(config, word_freq, token2id, pretrained_vectors, all_df):
    initial_vectors, unk_freqs = [], []
    for name, _pretrained_vectors in pretrained_vectors.items():
        vec, known_freq, unk_freq = build_word_vectors(
            word_freq, _pretrained_vectors, config['vocab']['min_count'])
        initial_vectors.append(vec)
        unk_freqs.append(unk_freq)
    initial_vectors = np.array(initial_vectors).mean(axis=0)
    unks = set.union(*[set(u.keys()) for u in unk_freqs])

    if config['embedding']['finetune']:
        tokens = all_df.tokens.values
        w2v = Word2Vec(size=300, min_count=1)
        w2v.build_vocab_from_freq(word_freq)
        idxmap = np.array([token2id[w] for w in w2v.wv.index2word])
        w2v.wv.vectors[:] = initial_vectors[idxmap]
        w2v.train(tokens, total_examples=len(tokens), epochs=3)
        initial_vectors = w2v.wv.vectors[idxmap.argsort()]

    initial_vectors[[token2id[u] for u in unks]] = 0
    embed = nn.Embedding.from_pretrained(
        torch.Tensor(initial_vectors), freeze=True)
    return embed, unk_freq


class Encoder(nn.Module):

    def __init__(self, config, embedding):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(config['embed']['dropout'])
        self.encoder = build_encoder(
            config['encoder']['name'])(config['encoder'])
        self.aggregator = build_aggregator(
            config['encoder']['aggregator'])
        self.attn = MultiHeadSelfAttention(
            attn_heads=4, in_size=config['encoder']['n_hidden'] * 2,
            out_size=config['encoder']['n_hidden'], dropout=0.)

    def forward(self, X, mask):
        h = self.embedding(X)
        h = self.dropout(h)
        h = self.encoder(h, mask)
        if self.attn is not None:
            h = self.attn(h, mask)
        h = self.aggregator(h, mask)
        return h
