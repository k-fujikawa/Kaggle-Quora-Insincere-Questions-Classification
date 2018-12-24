import torch
import numpy as np
from torch import nn

from qiqc.models import Word2VecEx


def build_sampler(i, epoch, weights):
    return None


def build_embedding(
        i, config, tokens, word_freq, token2id, pretrained_vectors):
    vecs = []
    for name, vec in pretrained_vectors.items():
        model = Word2VecEx(**config['embedding']['params'])
        model.build_vocab_from_freq(word_freq)
        model.initialize_pretrained_vector(vec)
        vecs.append(model.wv)

    # Fine tuning embedding
    model = Word2VecEx(**config['embedding']['params'])
    model.build_vocab_from_freq(word_freq)
    model.wv.vectors = np.array([v.wv.vectors for v in vecs]).mean(axis=0)
    if config['embedding']['finetune']:
        model.train(tokens, total_examples=len(tokens), epochs=1)
    mat = model.build_embedding_matrix(
        token2id, standardize=config['embedding']['standardize'])

    return mat


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
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        hidden_size = 60
        embed_size = 300
        max_features = 95000
        maxlen = 72

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

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


def build_model(i, config, embedding_matrix):
    clf = NeuralNet(embedding_matrix)
    return clf


def build_optimizer(i, config, model):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config['optimizer']['lr']))
    return optimizer
