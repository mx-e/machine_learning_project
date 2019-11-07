import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):

    def __init__(self, h, w, outputs):
        super(NN, self).__init__()

        linear_input_size = h * w
        self.fc1 = nn.Linear(linear_input_size, linear_input_size)
        self.fc2 = nn.Linear(linear_input_size, linear_input_size)
        self.head_value = nn.Linear(linear_input_size, 1)
        self.head_policy = nn.Linear(linear_input_size, outputs)
        self.head_policy_softmax = nn.Softmax()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x.view(x.size(0), -1)))
        value = self.head_value(x.view(x.size(0), -1))
        policy = self.head_policy(x.view(x.size(0), -1))
        policy = self.head_policy_softmax(policy.view(policy.size(0), -1))
        return value, policy
