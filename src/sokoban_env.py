import gym
import gym_sokoban
import numpy as np
import random
import math
import torch
import torchvision.transforms as T
from PIL import Image

class SokobanEnv:
    def __init__(self, env_mode = 'Sokoban-small-v1', eps_end = 0.05, eps_start = 0.95, eps_decay = 20000):
        self.env = gym.make(env_mode)

        self.resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = self.env.action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state, net):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return net(state).max(1)[1].view(1, 1).item()
        else:
            return self.env.action_space.sample()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()
        self.steps_done = 0
