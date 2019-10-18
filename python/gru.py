import math
import torch
import torch.nn.functional as F 

torch.manual_seed(42)

class GRUCell(torch.nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_features, state_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.bias = bias
        self.x2h = torch.nn.Linear(input_features, 3 * state_size, bias=bias)
        self.h2h = torch.nn.Linear(state_size, 3 * state_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.state_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, state):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(state)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (state - newgate)
        
        
        return hy