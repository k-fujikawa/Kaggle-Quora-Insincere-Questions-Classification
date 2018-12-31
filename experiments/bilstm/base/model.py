import torch
import numpy as np
from torch import nn

from qiqc.builder import build_aggregator
from qiqc.builder import build_encoder
from qiqc.models import EmbeddingUnit
from qiqc.models import BinaryClassifier


def build_sampler(i, epoch, weights):
    sampler = None
    # if epoch % 2 == 0:
    #     sampler = torch.utils.data.WeightedRandomSampler(
    #         weights=weights, num_samples=len(weights), replacement=True)
    # else:
    #     sampler = None
    return sampler


def build_embedding(
        i, config, tokens, unk_freq, token2id, initial_vectors):
    assert isinstance(initial_vectors, np.ndarray)
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
    return embed


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


def build_model(i, config, embedding):
    encoder = Encoder(config['model'], embedding)
    clf = BinaryClassifier(config['model'], encoder)
    return clf
