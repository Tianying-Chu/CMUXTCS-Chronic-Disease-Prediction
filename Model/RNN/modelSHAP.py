import torch
import torch.nn as nn


class rnnForward(nn.Module):

    def __init__(self):
        """ Initialize networks with default archetecture and weights """
        super(rnnForward, self).__init__()

        self.hidden = nn.Linear(269, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        o = torch.sigmoid(self.output(h))      
        return o
