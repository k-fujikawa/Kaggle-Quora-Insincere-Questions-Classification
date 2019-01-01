import torch
from torch import nn

from qiqc.models.fc.mlp import MLP


class BinaryClassifier(nn.Module):

    def __init__(self, config, encoder, lossfunc):
        super().__init__()
        self.encoder = encoder
        self.n_hidden = config['mlp']['n_hidden']
        in_size = config['encoder']['n_hidden'] *\
            config['encoder']['out_scale']
        self.mlp = MLP(
            n_layers=config['mlp']['n_layers'],
            in_size=in_size,
            out_size=config['mlp']['n_hidden'],
            actfun=nn.ReLU(True),
            bn0=config['mlp']['bn0'],
            bn=config['mlp']['bn'],
            dropout0=config['mlp']['dropout0'],
            dropout=config['mlp']['dropout'],
        )
        self.out = nn.Linear(config['mlp']['n_hidden'], 1)
        self.lossfunc = lossfunc

    def calc_loss(self, X, t, W=None):
        y = self.forward(X)
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

    def forward(self, X):
        h = self.predict_features(X)
        out = self.out(h)
        return out

    def predict_proba(self, X):
        y = self.forward(X)
        proba = torch.sigmoid(y).cpu().detach().numpy()
        return proba

    def predict_features(self, X):
        mask = X != 0
        maxlen = (mask == 1).any(dim=0).sum()
        X = X[:, :maxlen]
        mask = mask[:, :maxlen]

        h = self.encoder(X, mask)
        h = self.mlp(h)
        return h
