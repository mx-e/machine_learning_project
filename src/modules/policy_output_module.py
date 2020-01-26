import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from utils import initialize_weights


class Policy_Output_Module(nn.Module):  # an actor-critic neural network
    def __init__(self, input_size, num_action):
        super(Policy_Output_Module, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, num_action)
        self.apply(initialize_weights)
        self.input_size = input_size
        
    def forward(self, inputs, train=True, hard=False):
        
        inputs = inputs.view(-1, self.input_size)
        x = F.relu(self.linear1(inputs))
        return F.softmax(self.linear2(x), dim=1)