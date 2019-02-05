"""
Copyright (c) 2017-present, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

 * Neither the name Facebook nor the names of its contributors may be used to
    endorse or promote products derived from this software without specific
       prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""  # NOQA

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiqc.models import LSTMEncoder


class DynamicConvLSTMEncoder(LSTMEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.cnns = nn.ModuleList([])
        for param in config['params']:
            if param['kernel_size'] % 2 == 1:
                padding_l = param['kernel_size'] // 2
            else:
                padding_l = ((param['kernel_size'] - 1) // 2,
                             param['kernel_size'] // 2)
            cnn = DynamicConv1d(
                input_size=param['n_input'],
                kernel_size=param['kernel_size'],
                num_heads=param['num_heads'],
                padding_l=padding_l,
            )
            self.cnns.append(cnn)

    def forward(self, input, mask):
        h = input.transpose(0, 1).contiguous()
        for cnn in self.cnns:
            h = cnn(h)
            h = h.masked_fill(~mask.transpose(0, 1).unsqueeze(2), 0)
            h = h.contiguous()
        h = h.transpose(0, 1)
        out = super().forward(h, mask)
        return out


class DynamicConv1d(nn.Module):
    """Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """  # NOQA
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=True, conv_bias=True,
                 query_size=None, in_proj=False):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding

        if in_proj:
            self.weight_linear = nn.Linear(
                self.input_size, self.input_size + num_heads * kernel_size * 1)
        else:
            self.weight_linear = nn.Linear(
                self.query_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size \
            + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """  # NOQA
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(
                2, self.input_size, H*K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(x).view(T*B*H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(
                weight, self.weight_dropout, training=self.training,
                inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
            weight_expanded.as_strided(
                (B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(
                weight_expanded, self.weight_dropout, training=self.training,
                inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when
            # the kernel is larger than the length
            if K > T and P == K-1:
                weight = weight.narrow(2, K-T, T)
                K, P = T, T-1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(
                B*H, T, T+K-1, requires_grad=False)
            weight_expanded.as_strided(
                (B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)

        return output
