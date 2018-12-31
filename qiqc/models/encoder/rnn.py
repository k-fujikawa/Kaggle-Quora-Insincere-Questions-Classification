from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        rnns = []
        input_size = config['n_input']
        for i in range(config['n_layers']):
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=config['n_hidden'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                batch_first=True,
            )
            input_size = config['out_scale'] * config['n_hidden']
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)

    def forward(self, input):
        h = input
        for rnn in self.rnns:
            h, _ = rnn(h)
        h = h.contiguous()
        return h


class LSTMGRUEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        rnns = []
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
            input_size = config['out_scale'] * config['n_hidden']
            rnns.append(rnn)
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=config['n_hidden'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            batch_first=True,
        )
        rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)

    def forward(self, input):
        h = input
        for rnn in self.rnns:
            h, _ = rnn(h)
        h = h.contiguous()
        return h
