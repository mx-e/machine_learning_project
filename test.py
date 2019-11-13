import torch

from src import *

import warnings
warnings.filterwarnings('ignore')

NUM_EPISODES = 1

env = SokobanEnv()
init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = NN(screen_height * screen_width, n_actions).to(env.device)

snapshot = torch.load('./results/final_net')
performance_data = []



policy_net.load_state_dict(snapshot)
for x in range(20):
    snapshot_performance = dict()
    average_reward, average_ep_len = test_model(NUM_EPISODES, env, policy_net)
    snapshot_performance['avg_ep_len'] = average_ep_len
    snapshot_performance['avg_reward'] = average_reward
    snapshot_performance['episode'] = x
    print(f"{snapshot}: \nAVG GAME LEN:{average_ep_len}\nAVG_REWARD:{average_reward}")
    performance_data.append(snapshot_performance)
env.close()
print(performance_data)
#store_results_data(performance_data, s)
