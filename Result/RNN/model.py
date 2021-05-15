import torch
import torch.nn as nn


class nNet(nn.Module):

    def __init__(self):
        """ Initialize networks with default archetecture and weights """
        super(nNet, self).__init__()

        self.hidden = nn.Linear(260, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, x):
        h = torch.relu(self.hidden(x))
        o = torch.tanh(self.output(h))      
        return o
