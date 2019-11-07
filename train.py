import sys
import os
sys.path.append(os.getcwd())

from src import *

import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.80
EPS_END = 0
EPS_DECAY = 25000
TARGET_UPDATE = 10
MEM_SIZE = 25000
NUM_EPISODES = 200

env = SokobanEnv('Sokoban-small-v1', EPS_END, EPS_START, EPS_DECAY)

init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = NN(screen_height, screen_width, n_actions).to(env.device)
target_net = NN(screen_height, screen_width, n_actions).to(env.device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

dqn_optimizer = Optimizer(policy_net, target_net, env.device, BATCH_SIZE, MEM_SIZE, GAMMA)
train_model(dqn_optimizer, env, NUM_EPISODES, TARGET_UPDATE)
env.reset()
