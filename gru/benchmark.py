from __future__ import division
from __future__ import print_function

import argparse
import math
import time
import importlib

import pdb
import torch
from torch import nn

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda', 'torch_lib'])
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-s', '--state-size', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.gru_baseline import GRUCell 
elif options.example == 'torch_lib':
    i = "do nothing"
elif options.example == 'cpp':
    from cpp.gru import GRUCell
else:
    from cuda.gru import GRUCell
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs_hidden = {'dtype': dtype,
          'device': device,
          'requires_grad': True}

kwargs_input = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

X = torch.randn(options.batch_size, options.features, **kwargs_input)
h = torch.randn(options.batch_size, options.state_size, **kwargs_hidden)

if options.example == 'torch_lib':
    rnn = nn.GRUCell(options.features, options.state_size).to(device, dtype)
else:
    rnn = GRUCell(options.features, options.state_size).to(device, dtype)

# Force CUDA initialization
new_h = rnn(X, h)
new_h.sum().backward()
pdb.set_trace()

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0

for _ in range(options.runs):
    rnn.zero_grad()

    start = time.time()
    new_h = rnn(X, h)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    new_h.sum().backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: %.3f us, backward: %.3f us' % (forward_average, backward_average))
