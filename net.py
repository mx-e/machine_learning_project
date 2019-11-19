import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from utils import v_wrap, set_init, push_and_pull, record


class Net(nn.Module):
    def __init__(self, s_dim, a_dim, eps_start = 0.50, eps_end = 0.00, eps_decay=10000):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 512)
        self.pi2 = nn.Linear(512,128)
        self.pi3 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 256)
        self.v2 = nn.Linear(256,128)
        self.v3 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        pi2 = F.relu6(self.pi2(pi1))
        logits = self.pi3(pi2)
        v1 = F.relu6(self.v1(x))
        v2 = F.relu6(self.v2(v1))
        values = self.v3(v2)
        return logits, values

    def choose_action(self, s):

        if self.eps_start > 0.:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                self.eval()
                logits, _ = self.forward(s)
                prob = F.softmax(logits, dim=1).data
                m = self.distribution(prob)
                return m.sample().numpy()[0]
            else:
                return np.array(random.sample(range(self.a_dim), 1))
        else:
            self.eval()
            logits, _ = self.forward(s)
            prob = F.softmax(logits, dim=1).data
            m = self.distribution(prob)
            return m.sample().numpy()[0]


    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss