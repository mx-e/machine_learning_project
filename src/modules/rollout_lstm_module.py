import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from utils import initialize_weights


class Rollout_LSTM_Module(nn.Module):  # an actor-critic neural network
    def __init__(self, input_size, is_sokoban):
        super(Rollout_LSTM_Module, self).__init__()
        self.lstm = nn.LSTMCell(input_size, 256 if is_sokoban else 256)
        #self.apply(initialize_weights)

    def forward(self, inputs, train=True, hard=False):
        input, h0, c0 = inputs
        if(h0 is None or c0 is None):
            h1, c1 =  self.lstm(input)
        else:
            h1, c1 = self.lstm(input, (h0, c0))
        return h1, c1

