import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from utils import initialize_weights


class Conv2d_Module(nn.Module):  # an actor-critic neural network
    def __init__(self, is_sokoban):
        super(Conv2d_Module, self).__init__()
        if(is_sokoban):
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.apply(initialize_weights)

    def forward(self, inputs, train=True, hard=False):
        x = F.relu(self.conv1(inputs))
        return F.relu(self.conv2(x))


