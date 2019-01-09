import torch
import numpy as np
from gensim.models import Word2Vec
from torch import nn

from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.embeddings import build_word_vectors
from qiqc.models import EmbeddingUnit
from qiqc.models import BinaryClassifier


def build_models(config, word_freq, token2id, pretrained_vectors, all_df):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    if config['embedding']['finetune']:
        initial_vectors, unk_freq = build_embedding(
            config, word_freq, pretrained_vectors)
        finetuned_vectors = finetune_embedding(
            config, word_freq, token2id, initial_vectors, all_df)
        unk_ids = np.array([
            token2id[token] for token, freq in unk_freq.items()
            if freq < config['vocab']['min_count']])
        finetuned_vectors[0] = 0
        if len(unk_ids) > 0:
            finetuned_vectors[unk_ids] = 0

    for i in range(config['cv']):
        initial_vectors, unk_freq = build_embedding(
            config, word_freq, pretrained_vectors)
        if config['embedding']['finetune']:
            initial_vectors = (initial_vectors + finetuned_vectors) / 2
        embedding = nn.Embedding.from_pretrained(
            torch.Tensor(initial_vectors), freeze=True)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)

    return models, unk_freq


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


def build_embedding(config, word_freq, pretrained_vectors):
    initial_vectors, unk_freqs = [], []
    for name, _pretrained_vectors in pretrained_vectors.items():
        vec, known_freq, unk_freq = build_word_vectors(
            word_freq, _pretrained_vectors, config['vocab']['min_count'])
        initial_vectors.append(vec)
        unk_freqs.append(unk_freq)
    initial_vectors = np.array(initial_vectors).mean(axis=0)
    return initial_vectors, unk_freq


def finetune_embedding(config, word_freq, token2id, initial_vectors, all_df):
    tokens = all_df.tokens.values
    w2v = Word2Vec(size=300, min_count=1, workers=1)
    w2v.build_vocab_from_freq(word_freq)
    idxmap = np.array([token2id[w] for w in w2v.wv.index2word])
    w2v.wv.vectors[:] = initial_vectors[idxmap]
    w2v.trainables.syn1neg[:] = initial_vectors[idxmap]
    w2v.train(tokens, total_examples=len(tokens), epochs=5)
    finetuned_vectors = w2v.wv.vectors[idxmap.argsort()]
    return finetuned_vectors


class Encoder(nn.Module):

    def __init__(self, config, embedding):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(config['embed']['dropout'])
        self.encoder = build_encoder(
            config['encoder']['name'])(config['encoder'])
        self.aggregator = build_aggregator(
            config['encoder']['aggregator'])

    def forward(self, X, mask):
        h = self.embedding(X)
        h = self.dropout(h)
        h = self.encoder(h, mask)
        h = self.aggregator(h, mask)
        return h
