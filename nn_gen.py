import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class BasicNet(nn.Module):
    def __init__(self, n_layers=8, neurons=12):
        super(BasicNet, self).__init__()
        intermediate_layer_list = []
        for i in range(n_layers):
            intermediate_layer_list.append(nn.Linear(neurons, neurons, bias=False))
            intermediate_layer_list.append(nn.Tanh())
        self.layers = nn.Sequential(nn.Linear(2, neurons, bias=True),
                                    nn.Tanh(),
                                    *intermediate_layer_list,
                                    nn.Linear(neurons, 3, bias=False),
                                    nn.Sigmoid())
        self.apply(init_normal)

    def forward(self, x):
        return self.layers(x)

