
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_size, n_hiddens, actfun=nn.ReLU(True),
                 bn=True, dropout=0., bn0=False, dropout0=0.):
        super().__init__()
        assert isinstance(n_hiddens, list)
        layers = []
        if bn0:
            layers.append(nn.BatchNorm1d(in_size))
        if dropout0 > 0:
            layers.append(nn.Dropout(dropout0))
        for n_hidden in n_hiddens:
            layers.append(nn.Linear(in_size, n_hidden))
            if actfun is not None:
                layers.append(actfun)
            if bn:
                layers.append(nn.BatchNorm1d(n_hidden))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = n_hidden
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
