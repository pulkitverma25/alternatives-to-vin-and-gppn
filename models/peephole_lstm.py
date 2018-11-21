import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class PLSTMCell(nn.Module): 
    def __init__(self, input_size, hidden_size, cell_size, bias=True):
        super(PLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_ch = Parameter(torch.Tensor(hidden_size, cell_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
            self.bias_ch = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def LSTMPCell(self, input, hidden, cell, w_ih, w_hh, w_ch, b_ih=None, b_hh=None, b_ch=None):
        '''
        if input.is_cuda:
            igates = F.linear(input, w_ih)
            hgates = F.linear(hidden[0], w_hh)
            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        '''
        hx = hidden
        cx = cell
        input_gates = F.linear(input, w_ih, b_ih)
        hidden_gates = F.linear(hx, w_hh, b_ch)
        cell_gates = F.linear(cx, w_ch, b_ch)

        ingate = input_gates + hidden_gates + cell_gates
        ingate = torch.sigmoid(ingate)

        forgetgate = input_gates + hidden_gates + cell_gates
        forgetgate = torch.sigmoid(forgetgate)

        cellgate = input_gates + hidden_gates 
        cellgate = torch.tanh(cellgate)
        cy = (forgetgate * cx) + (ingate * cellgate)

        outgate = input_gates + hidden_gates + F.linear(cy, w_ch, b_ch)
        outgate = torch.sigmoid(outgate)

        hy = outgate * torch.tanh(cy)

        return hy, cy

    def forward(self, input, hx, cx):
        return self.LSTMPCell(
            input, hx, cx,
            self.weight_ih, self.weight_hh, self.weight_ch,
            self.bias_ih, self.bias_hh, self.bias_ch,
        )



