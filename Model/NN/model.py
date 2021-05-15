# -*- coding: utf-8 -*-
"""
Created on Fri April 16 21:03:29 2021

@author: Yilun Chen
"""
import sys
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class nNet(nn.Module):
    
    def __init__(self):
        super(nNet, self).__init__()
        nHidden = 20
        self.hidden = nn.Linear(248, nHidden)
        self.output = nn.Linear(nHidden, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        o = torch.sigmoid(self.output(h))
        return o

def learnNN(lambda_l2, epoch, lr, momentum, xTrain, yTrain, load_model, model_type):
    net = nNet().double()
    if load_model==True:
      net.load_state_dict(torch.load('Model/NN/{}'.format(model_type)))
    else:
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0)      
      for e in range(epoch):
        outTrain = net(xTrain).reshape(-1)
        criterion = nn.MSELoss()
        lossTrain = criterion(outTrain, yTrain) 
        l2 = 0.00
        for p in net.parameters():
          l2 = l2+torch.square((p.abs())).sum()
        lossTrainTotal = criterion(outTrain, yTrain) + lambda_l2 * l2
        optimizer.zero_grad()
        lossTrainTotal.backward()
        optimizer.step()
    return net
