import torch
import numpy as np
from torch import nn

from qiqc.embeddings import build_word_vectors
from qiqc.models import EmbeddingUnit


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
    clf = NeuralNet(embedding)
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


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self, embedding):
        super(NeuralNet, self).__init__()

        hidden_size = 60
        embed_size = 300
        maxlen = 72

        self.embedding = embedding

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(
            hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size*2, maxlen)
        self.gru_attention = Attention(hidden_size*2, maxlen)

        self.linear = nn.Linear(480, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def calc_loss(self, X, t, W=None):
        y = self.forward(X)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        loss = loss_fn(y, t)
        output = dict(
            y=torch.sigmoid(y).cpu().data.numpy(),
            t=t.cpu().data.numpy(),
            loss=loss.cpu().data.numpy(),
        )
        return loss, output

    def predict_proba(self, X):
        y = self.forward(X)
        proba = torch.sigmoid(y).cpu().detach().numpy()
        return proba

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out

    def to_device(self, device):
        self.device = device
        self.to(device)
        return self
