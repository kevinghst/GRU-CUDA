import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function

import math


cuda = True if torch.cuda.is_available() else False
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
 
test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
 
batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


def d_sigmoid(s):
    return (1 - s) * s


def d_tanh(t):
    return 1 - (t * t)

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

        ctx.save_for_backward(resetgate, inputgate, newgate, h_n, old_h, x, x2h_w, h2h_w)
        
        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):
        rg, ig, ng, hn, hx, x, x2h_w, h2h_w = ctx.saved_variables[:8]
        

        grad_x = grad_input_gates = grad_hidden_gates = grad_input_bias = grad_hidden_bias = grad_hx = None

        grad_hx = grad_hy * ig

        gig = d_sigmoid(ig) * grad_hy * (hx - ng)
        gin = d_tanh(ng) * grad_hy * (1 - ig)
        ghn = gin * rg
        grg = d_sigmoid(rg) * gin * hn 

        grad_input_gates = torch.cat(
            [grg, gig, gin], dim=1
        )
        grad_hidden_gates = torch.cat(
            [grg, gig, ghn], dim=1
        )

        grad_input_weights = grad_input_gates.t().mm(x)
        grad_hidden_weights = grad_hidden_gates.t().mm(hx)

        grad_input_bias = grad_input_gates.sum(dim=0, keepdim=True)
        grad_hidden_bias = grad_hidden_gates.sum(dim=0, keepdim=True)

        # grad_x = grad_input_gates.mm(x2h_w)
        grad_hx += grad_hidden_gates.mm(h2h_w)

        return grad_x, grad_input_weights, grad_hidden_weights, grad_input_bias, grad_hidden_bias, grad_hx


class GRUCell(torch.nn.Module):

    #An implementation of GRUCell.


    def __init__(self, input_features, state_size, bias=True):
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

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
         
       
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
         
       
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)
            

        out = outs[-1].squeeze()
        
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out
 
#STEP 4: INSTANTIATE MODEL CLASS

input_dim = 28
hidden_dim = 128
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10
 
# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################
 
if torch.cuda.is_available():
    model.cuda()
     

#STEP 5: INSTANTIATE LOSS CLASS

criterion = nn.CrossEntropyLoss()
 

#STEP 6: INSTANTIATE OPTIMIZER CLASS

learning_rate = 0.1
 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#STEP 7: TRAIN THE MODEL

 
# Number of steps to unroll
seq_dim = 28 

loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        #######################
        #  USE GPU FOR MODEL  #
        #######################
          
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)
          
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            loss.cuda()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        
        loss_list.append(loss.item())
        iter += 1
         
        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1 , seq_dim, input_dim))
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                 
                # Total number of labels
                total += labels.size(0)
                 
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
             
            accuracy = 100 * correct / total
             
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))