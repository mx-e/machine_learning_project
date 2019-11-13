import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):

    def __init__(self, input_size, output_size):
        super(NN, self).__init__()

        self.linear_input_size = input_size
        self.policy_output_size = output_size

        self.policy1 = nn.Linear(self.linear_input_size, 200)
        self.policy2 = nn.Linear(200, 200)
        self.policy_head = nn.Linear(200, self.policy_output_size)

        self.value1 = nn.Linear(self.linear_input_size, 200)
        self.value_head = nn.Linear(200, 1)

        self.set_init([self.policy1, self.policy_head, self.value1, self.value_head])
        self.distribution = torch.distributions.Categorical

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        policy = F.relu6(self.policy1(x))
        policy = F.relu6(self.policy2(policy))
        policy = self.policy_head(policy)

        value = F.relu6(self.value1(x))
        value = self.value_head(value)

        return value, policy

    def choose_action(self, s):
        self.eval()
        _, policy = self.forward(s)
        prob = F.softmax(policy, dim=0).data
        m = self.distribution(prob)
        return m.sample().item()

    def loss_func(self, s, a, v_t):
        self.train()
        values, policy  = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(policy, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)