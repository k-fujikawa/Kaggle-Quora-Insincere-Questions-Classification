import torch
from torch import nn


class BinaryClassifier(nn.Module):

    default_config = None

    def __init__(self, embedding, encoder, aggregator, mlp, out, lossfunc):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.aggregator = aggregator
        self.mlp = mlp
        self.out = out
        self.lossfunc = lossfunc

    def calc_loss(self, X, X2, t, W=None):
        y = self.forward(X, X2)
        loss = self.lossfunc(y, t)
        output = dict(
            y=torch.sigmoid(y).cpu().detach().numpy(),
            t=t.cpu().detach().numpy(),
            loss=loss.cpu().detach().numpy(),
        )
        return loss, output

    def to_device(self, device):
        self.device = device
        self.to(device)
        return self

    def forward(self, X, X2):
        h = self.predict_features(X, X2)
        out = self.out(h)
        return out

    def predict_proba(self, X, X2):
        y = self.forward(X, X2)
        proba = torch.sigmoid(y).cpu().detach().numpy()
        return proba

    def predict_features(self, X, X2):
        mask = X != 0
        maxlen = (mask == 1).any(dim=0).sum()
        X = X[:, :maxlen]
        mask = mask[:, :maxlen]

        h = self.embedding(X)
        h = self.encoder(h, mask)
        h = self.aggregator(h, mask)
        h = self.mlp(h, X2)
        return h
