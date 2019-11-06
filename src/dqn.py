import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        linear_input_size = h * w
        self.fc1 = nn.Linear(linear_input_size, linear_input_size)
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))
