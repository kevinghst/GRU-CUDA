import math
from torch import nn
from torch.autograd import Function
import torch

import gru_cpp

torch.manual_seed(42)

class GRUFunction(Function):
    @staticmethod
    def forward(ctx, input, x2h_w, h2h_w, x2h_b, h2h_b, old_h):
        outputs = gru_cpp.forward(input, x2h_w, h2h_w, x2h_b, h2h_b, old_h)
        new_h = outputs[0]
        # Backward stuff
        return new_h

class GRUCell(nn.Module):
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
        input = input.view(-1, input.size(1))

        return GRUFunction.apply(
            input, 
            self.x2h_weights, self.h2h_weights,
            self.x2h_bias, self.h2h_bias,
            state    
        )