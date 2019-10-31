from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.gru_baseline import GRUFunction
elif options.example == 'cpp':
    from cpp.gru import GRUFunction
else:
    from cuda.gru import GRUFunction
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs_hidden = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

kwargs_input = {'dtype': torch.float64,
          'device': device,
          'requires_grad': False}

X = torch.randn(options.batch_size, options.features, **kwargs_input)
h = torch.randn(options.batch_size, options.state_size, **kwargs_hidden)

Wx = torch.randn(3 * options.state_size, options.features, **kwargs_hidden)
Wh = torch.randn(3 * options.state_size, options.state_size, **kwargs_hidden)
bx = torch.randn(1, 3 * options.state_size, **kwargs_hidden)
bh = torch.randn(1, 3 * options.state_size, **kwargs_hidden)

variables = [X, Wx, Wh, bx, bh, h]


if gradcheck(GRUFunction.apply, variables):
    print('Ok')