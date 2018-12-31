from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        rnns = []
        dropouts = []
        input_size = config['n_input']
        for i in range(config['n_layers']):
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=config['n_hidden'],
                bidirectional=config['bidirectional'],
                batch_first=True,
            )
            dropout = nn.Dropout(config['dropout'])
            input_size = config['out_scale'] * config['n_hidden']
            rnns.append(rnn)
            dropouts.append(dropout)
        self.rnns = nn.ModuleList(rnns)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, input):
        h = input
        for rnn, dropout in zip(self.rnns, self.dropouts):
            h, _ = rnn(h)
            h = dropout(h)
        h = h.contiguous()
        return h


class LSTMGRUEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        rnns = []
        dropouts = []
        input_size = config['n_input']
        assert config['n_layers'] >= 2
        for i in range(config['n_layers'] - 1):
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=config['n_hidden'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                batch_first=True,
            )
            dropout = nn.Dropout(config['dropout'])
            input_size = config['out_scale'] * config['n_hidden']
            rnns.append(rnn)
            dropouts.append(dropout)
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=config['n_hidden'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            batch_first=True,
        )
        rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, input):
        h = input
        for rnn in self.rnns:
            h, _ = rnn(h)
        h = h.contiguous()
        return h
