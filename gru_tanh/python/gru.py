import pdb

import math
import torch
import torch.nn.functional as F 
from torch import nn

torch.manual_seed(42)

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
    
    def forward(self, x, state):
        x = x.view(-1, x.size(1))
        
        gate_x = F.linear(x, self.x2h_weights, self.x2h_bias)
        gate_h = F.linear(state, self.h2h_weights, self.h2h_bias)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.tanh(i_r + h_r)
        inputgate = F.tanh(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (state - newgate)
        
        
        return hy