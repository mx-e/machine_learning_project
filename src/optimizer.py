
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from .replay_memory import ReplayMemory
from .transition import Transition

class Optimizer:
    def __init__(self, policy_net, target_net, device, batch_size = 128, memory_size = 25000, gamma = 0.95):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma

    def optimize_model(self, states, actions, rewards):
        R = 0
        for j in range(len(states) - 1, -1, -1):
            R = rewards[j] + self.gamma * R
            val = self.policy_net(states[j])[0]

            actual_pol = self.policy_net(states[j])[1][0][actions[j]]

            criterion1 = nn.L1Loss()
            loss1 = criterion1(actual_pol * val, actual_pol * R)
            loss1.backward(retain_graph=True)
            criterion2 = nn.MSELoss()
            loss2 = criterion2(val, torch.tensor(R))
            loss2.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        print("score", R)
        print('actions', actions)
