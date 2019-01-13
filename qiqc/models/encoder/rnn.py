import torch
from torch import nn


class RNNEncoderBase(nn.Module):

    def __init__(self, config, modules, pack=False, mask=False):
        super().__init__()
        self.pack = pack
        self.mask = mask
        rnns = []
        input_size = config['n_input']
        for module in modules:
            rnn = module(
                input_size=input_size,
                hidden_size=config['n_hidden'],
                bidirectional=config['bidirectional'],
                batch_first=True,
            )
            n_direction = int(config['bidirectional']) + 1
            input_size = n_direction * config['n_hidden']
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)

    def forward(self, input, mask):
        h = input
        if self.pack:
            seq_lengths, perm_idx = mask.sum(dim=1).sort(0, descending=True)
            h = torch.nn.utils.rnn.pack_padded_sequence(
                h[perm_idx], seq_lengths, batch_first=True)
        for rnn in self.rnns:
            if self.mask:
                _mask = mask[:, :, None].repeat(1, 1, h.shape[-1])
                fill_value = torch.full(h.shape, 0).to(h.device)
                h = torch.where(_mask, h, fill_value)
            h, _ = rnn(h)

        if self.pack:
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(
                h, batch_first=True)
            h = h[perm_idx.argsort()]
        return h


class LSTMEncoder(RNNEncoderBase):

    def __init__(self, config):
        modules = [nn.LSTM] * config['n_layers']
        super().__init__(config, modules)


class GRUEncoder(RNNEncoderBase):

    def __init__(self, config):
        assert config['n_layers'] > 1
        modules = [nn.GRU] * config['n_layers']
        super().__init__(config, modules)


class LSTMGRUEncoder(RNNEncoderBase):

    def __init__(self, config):
        assert config['n_layers'] > 1
        modules = [nn.LSTM] * (config['n_layers'] - 1) + [nn.GRU]
        super().__init__(config, modules)


class GRULSTMEncoder(RNNEncoderBase):

    def __init__(self, config):
        assert config['n_layers'] > 1
        modules = [nn.GRU] * (config['n_layers'] - 1) + [nn.LSTM]
        super().__init__(config, modules)
