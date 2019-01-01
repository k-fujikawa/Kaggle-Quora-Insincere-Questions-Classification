
from torch import nn


class MLP(nn.Module):

    def __init__(self, n_layers, in_size, out_size, actfun=nn.ReLU(True),
                 bn=True, dropout=0., bn0=False, dropout0=0.):
        super().__init__()
        layers = []
        if bn0:
            layers.append(nn.BatchNorm1d(in_size))
        if dropout0 > 0:
            layers.append(nn.Dropout(dropout0))
        for i in range(n_layers):
            layers.append(nn.Linear(in_size, out_size))
            if actfun is not None:
                layers.append(actfun)
            if bn:
                layers.append(nn.BatchNorm1d(out_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = out_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
