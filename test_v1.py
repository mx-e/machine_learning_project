import torch

from src.dqn import DQN
from src.sokoban_env import SokobanEnv
from src.test_loop import test_model

import os
if not os.path.exists("model"):
    os.mkdir("model")

import warnings
warnings.filterwarnings('ignore')

NUM_EPISODES = 10

env = SokobanEnv()
init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(env.device)

snapshots = torch.load('./model/snapshots')
for snapshot in snapshots:
    print(snapshot)
for snapshot in snapshots:
    policy_net.load_state_dict(snapshots[snapshot])
    print(test_model(NUM_EPISODES, env, policy_net))

env.close()
