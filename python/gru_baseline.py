import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

torch.manual_seed(42)

class GRUFunction(Function):
    @staticmethod

    def forward(ctx, input, x2h_w, h2h_w, x2h_b, h2h_b, old_h):
        x = input.view(-1, input.size(1))

        gate_x = F.linear(x, x2h_w, x2h_b)
        gate_h = F.linear(old_h, h2h_w, h2h_b)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (old_h - newgate)
        
        # add backward stuff
        
        return hy



class GRUCell(torch.nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_features, state_size):
        super(GRUCell, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.x2h_weights = nn.Parameter(torch.Tensor(3 * state_size, input_features))
        self.h2h_weights = nn.Parameter(torch.Tensor(3 * state_size, state_size))
        self.x2h_bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.h2h_bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.state_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state):
        return GRUFunction.apply(
            input, self.x2h_weights, self.h2h_weights, self.x2h_bias, self.h2h_bias, state
        )