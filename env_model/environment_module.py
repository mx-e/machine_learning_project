
import sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../src')
from utils import initialize_weights, Normalize


class Env_Module(nn.Module):  # an actor-critic neural network
    def __init__(self, input_size, num_actions):
        super(Env_Module, self).__init__()
        self.action_space = num_actions
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=self.action_space + 1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.deconf1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.val_conv1 = self._make_stage(32, 32)
        self.val_conv2 = self._make_stage(32, 1)
        self.val_linear = nn.Linear(25, 5)
        self.val_output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.cuda = torch.cuda.device_count() > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(initialize_weights)

    def forward(self, inputs, train=True, hard=False):
        state, action = inputs

        action_one_hot_encoding = torch.tensor((self.action_space), device=self.device, dtype=torch.float)
        action_one_hot_encoding = action_one_hot_encoding.new_zeros((self.action_space))
        action_one_hot_encoding.index_fill_(0, action, 1)
        action_one_hot_encoding = action_one_hot_encoding.view(-1,1,1).repeat(self.input_size)
        model_input = torch.cat([state, action_one_hot_encoding]).unsqueeze(0)

        x = F.relu(self.conv1(model_input))
        x = F.relu(self.conv2(x))
        predicted_state = self.sigmoid(self.deconf1(x))

        x = self.val_conv1(x)
        x = self.val_conv2(x)
        x = x.view(-1, 25)
        x = F.softmax(self.val_linear(x), dim=1)
        predicted_value = self.val_output(x).squeeze(0)

        return predicted_state, predicted_value

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar');
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) if not 'production' in s  else 0 for s in paths]
            ix = np.argmax(ckpts);
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

    def _make_stage(self, in_channels, out_channels):
        stage = nn.Sequential()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
                )

        stage.add_module('conv', conv)
        stage.add_module('relu', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=1))
        return stage