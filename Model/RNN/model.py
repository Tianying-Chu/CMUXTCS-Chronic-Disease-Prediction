import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class RecurrentNet(nn.Module):

    def __init__(self, nHidden):
        """ Initialize networks with default archetecture and weights """
        super(RecurrentNet, self).__init__()

        self.rnn = nn.RNN(
            input_size=248,
            hidden_size=nHidden,
            num_layers=1,
            nonlinearity='tanh',
            bias=True
        )
        self.output = nn.Linear(nHidden, 1) #(3,N,20)
        self.recurr = nn.Linear(1, 1)

    def forward(self, x):
        o = []
        oFinal = torch.zeros((x.shape[1], x.shape[0]))
        h = self.rnn(x)[0]  
        o.append(torch.sigmoid(self.output(h[0])))
        for i in range(x.shape[0]-1):
          o.append(torch.sigmoid(self.output(h[i+1])+self.recurr(o[i])))  
        for i in range(len(o)):
          oFinal[:,i] = o[i].reshape(-1)
        return oFinal

class nNet(nn.Module):

    def __init__(self, nHidden, l0, l1, b0, b1):
        """ Initialize networks with default archetecture and weights """
        super(nNet, self).__init__()

        self.hidden = nn.Linear(269, nHidden)
        self.output = nn.Linear(nHidden, 1)

        with torch.no_grad():
            self.hidden.weight = torch.nn.Parameter(l0)
            self.output.weight = l1
            self.hidden.bias = torch.nn.Parameter(b0)
            self.output.bias = b1

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        o = torch.sigmoid(self.output(h))      
        return o

def learnRNN(lambda_l2, nHidden, epoch, lr, momentum, xTrain, yTrain, load_model, model_type):
    net = RecurrentNet(nHidden).double()
    if load_model==True:
        net.load_state_dict(torch.load('Model/RNN/{}'.format(model_type)))
    if load_model==False: 
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0)
        for e in range(epoch):
            outTrain = net(xTrain)
            criterion = nn.MSELoss()
            lossTrain = criterion(outTrain[:,-1].reshape(-1), yTrain[-1])
            l2 = 0.00
            for p in net.parameters():
              l2 = l2 + torch.square((p.abs())).sum()
      
            lossTrainTotal = 0.00
            for i in range(outTrain.shape[1]):
               lossTrainTotal += criterion(outTrain[:,i].reshape(-1).float(), yTrain[i].float())
            lossTrainTotal += lambda_l2 * l2
      
            optimizer.zero_grad()
            lossTrainTotal.backward()
            optimizer.step()
    return net
