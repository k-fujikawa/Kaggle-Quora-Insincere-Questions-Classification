from torch import nn

from qiqc.registry import register_encoder
from qiqc.registry import AGGREGATOR_REGISTRY


class RNNEncoderBase(nn.Module):

    def __init__(self, config, modules, in_size):
        super().__init__()
        rnns = []
        input_size = in_size
        for module in modules:
            rnn = module(
                input_size=input_size,
                hidden_size=config.encoder_n_hidden,
                bidirectional=config.encoder_bidirectional,
                batch_first=True,
            )
            n_direction = int(config.encoder_bidirectional) + 1
            input_size = n_direction * config.encoder_n_hidden
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)
        self.out_size = n_direction * config.encoder_n_hidden

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--encoder-bidirectional', type=bool, default=True)
        parser.add_argument('--encoder-dropout', type=float, default=0.)
        parser.add_argument('--encoder-n-hidden', type=int)
        parser.add_argument('--encoder-n-layers', type=int)
        parser.add_argument('--encoder-aggregator', type=str,
                            choices=AGGREGATOR_REGISTRY)

    def forward(self, input, mask):
        h = input
        for rnn in self.rnns:
            h, _ = rnn(h)
        return h


@register_encoder('lstm')
class LSTMEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        modules = [nn.LSTM] * config.encoder_n_layers
        super().__init__(config, modules, in_size)


@register_encoder('gru')
class GRUEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.GRU] * config.encoder_n_layers
        super().__init__(config, modules, in_size)


@register_encoder('lstmgru')
class LSTMGRUEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.LSTM] * (config.encoder_n_layers - 1) + [nn.GRU]
        super().__init__(config, modules, in_size)


@register_encoder('grulstm')
class GRULSTMEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.GRU] * (config.encoder_n_layers - 1) + [nn.LSTM]
        super().__init__(config, modules, in_size)
