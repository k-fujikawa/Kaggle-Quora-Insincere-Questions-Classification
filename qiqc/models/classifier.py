from torch import nn

import qiqc


class BinaryClassifier(nn.Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.mlp = qiqc.models.MLP(
            n_layers=config['mlp']['n_layers'],
            in_size=config['encoder']['n_hidden'] * \
                config['encoder']['out_scale'],
            out_size=config['mlp']['n_hidden'],
            actfun=nn.ReLU(True),
            bn=config['mlp']['bn'],
        )
        self.out = nn.Linear(config['mlp']['n_hidden'], 1)
        self.lossfunc = nn.BCEWithLogitsLoss()

    def calc_loss(self, batch):
        X, t, mask = batch['token_ids'], batch['target'], batch['mask']
        y = self.forward(X, mask).view(-1)
        loss = self.lossfunc(y, t)
        return loss, y

    def to_device(self, device):
        self.device = device
        self.to(device)
        return self

    def forward(self, X, mask):
        h = self.encoder(X, mask)
        h = self.mlp(h)
        out = self.out(h)
        return out
