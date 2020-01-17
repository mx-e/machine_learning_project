import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from utils import initialize_weights


class Linear_Module(nn.Module):  # an actor-critic neural network
    def __init__(self, input_size, num_action, is_sokoban):
        super(Linear_Module, self).__init__()
        if(is_sokoban):
            self.linear = nn.Linear(input_size, 256)
            self.logit = nn.Linear(256, num_action)
            self.value = nn.Linear(256, 1)
        else:
            self.linear = nn.Linear(input_size, 256)
            self.logit = nn.Linear(256, num_action)
            self.value = nn.Linear(256, 1)

        self.apply(initialize_weights)

    def forward(self, inputs, train=True, hard=False):
        x = F.relu(self.linear(inputs))
        return (F.softmax(self.logit(x), dim=1), self.value(x))