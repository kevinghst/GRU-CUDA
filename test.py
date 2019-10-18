from __future__ import division
from __future__ import print_function

import pdb

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda', 'py_baseline'])
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-s', '--state-size', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.gru import GRUCell
elif options.example == 'py_baseline':
    from python.gru_baseline import GRUCell
elif options.example == 'cpp':
    from cpp.gru import GRUCell
else:
    from cuda.gru import GRUCell
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size, options.features, **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)

rnn = GRUCell(options.features, options.state_size).to(device, dtype)

# Force CUDA initialization
new_h = rnn(X, h)
# (new_h.sum() + new_C.sum()).backward()

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0

rnn.zero_grad()
new_h = rnn(X, h)

pdb.set_trace()

print("done")