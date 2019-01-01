import torch
import numpy as np
from torch import nn

from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.embeddings import build_word_vectors
from qiqc.models import EmbeddingUnit
from qiqc.models import BinaryClassifier


def build_models(config, word_freq, token2id, pretrained_vectors):
    models = []
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(config['device'])
    for i in range(config['cv']):
        embedding, unk_freq = build_embedding(
            config, word_freq, token2id, pretrained_vectors)
        lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = build_model(config, embedding, lossfunc)
        models.append(model)
    return models, unk_freq


def build_model(config, embedding, lossfunc):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder, lossfunc)
    return clf


def build_embedding(config, word_freq, token2id, pretrained_vectors):
    initial_vectors, unk_freqs = [], []
    for name, _pretrained_vectors in pretrained_vectors.items():
        vec, known_freq, unk_freq = build_word_vectors(
            word_freq, _pretrained_vectors, config['vocab']['min_count'])
        initial_vectors.append(vec)
        unk_freqs.append(unk_freq)
    initial_vectors = np.array(initial_vectors).mean(axis=0)

    if config['embedding']['finetune']:
        unfixed_tokens = set([token for token, freq in unk_freq.items()
                             if freq >= config['vocab']['min_count']])
        fixed_idxmap = [idx if token not in unfixed_tokens else 0
                        for token, idx in token2id.items()]
        unfixed_idxmap = [idx if token in unfixed_tokens else 0
                          for token, idx in token2id.items()]
        fixed_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(initial_vectors[fixed_idxmap]), freeze=True)
        unfixed_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(initial_vectors[unfixed_idxmap]), freeze=False)
        embed = EmbeddingUnit(fixed_embedding, unfixed_embedding)
    else:
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

    def forward(self, X, mask):
        h = self.embedding(X)
        h = self.dropout(h)
        h = self.encoder(h)
        h = self.aggregator(h, mask)
        return h
