from torch import nn


class EmbeddingWrapper(nn.Module):

    def __init__(self, embedding, dropout1d=0., dropout2d=0., noise=False):
        super().__init__()
        self.embedding = embedding
        self.dropout1d = nn.Dropout(dropout1d)
        self.dropout2d = nn.Dropout2d(dropout2d)
        self.noise = noise

    def forward(self, x):
        h = self.embedding(x)
        h = self.dropout1d(h)
        h = self.dropout2d(h)

        return h
