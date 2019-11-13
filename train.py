import sys
import os
import torch.multiprocessing as mp
import plotly.express as px
sys.path.append(os.getcwd())
import torch

from src import *

import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 50000
TARGET_UPDATE = 100
MEM_SIZE = 25000
NUM_EPISODES = 9999999

env = SokobanEnv('Sokoban-small-v1', EPS_END, EPS_START, EPS_DECAY)
input_size = len(env.get_screen())
output_size = env.action_space.n
# Get number of actions from gym action space

gnet = NN(input_size, output_size)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [AsyncTrainingWorker(gnet, opt, global_ep, global_ep_r, res_queue, i, input_size, output_size) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot
while True:
    r = res_queue.get()
    if r is not None:
        res.append(r)
    else:
        break
[w.join() for w in workers]

import matplotlib.pyplot as plt
plt.plot(res)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.show()
torch.save(gnet.state_dict(), 'results/final_net' )

env.reset()
