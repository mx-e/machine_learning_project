import torch

from src.dqn import DQN
from src.sokoban_env import SokobanEnv
from src.test_loop import test_model
from src.store_results_data import store_results_data

import warnings
warnings.filterwarnings('ignore')

NUM_EPISODES = 10

env = SokobanEnv()
init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(env.device)

snapshots = torch.load('./results/snapshots')
performance_data = []

print('SNAPSHOTS: ')
for snapshot in snapshots:
    print(snapshot)

for snapshot in snapshots:
    snapshot_performance = dict()
    policy_net.load_state_dict(snapshots[snapshot])
    average_reward, average_ep_len = test_model(NUM_EPISODES, env, policy_net)
    snapshot_performance['avg_ep_len'] = average_ep_len
    snapshot_performance['avg_reward'] = average_reward
    snapshot_performance['episode'] = int(snapshot.split('@')[1])
    print(f"{snapshot}: \nAVG GAME LEN:{average_ep_len}\nAVG_REWARD:{average_reward}")
    performance_data.append(snapshot_performance)
env.close()
store_results_data(performance_data, snapshots)
