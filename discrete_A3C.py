
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import gym_sokoban
import os

from shared_adam import SharedAdam
from shared_rms_prop import SharedRMSProp
from net import Net
from sokoban_env import SokobanEnv
from async_training_worker import AsyncTrainingWorker

os.environ["OMP_NUM_THREADS"] = "1"

GAMMA = 0.99
MAX_EP = 500000
UPDATE_GLOBAL = 50
ADAM = True 
SOKOBAN = False
# alternative ENV here
ENV = 'SpaceInvaders-ram-v0'



if(SOKOBAN):
    env = SokobanEnv()
else:
    env = gym.make(ENV)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


gnet = Net(N_S, N_A)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
if(ADAM):
    opt = SharedAdam(gnet.parameters(), lr=0.001)
else:
    opt = SharedRMSProp(gnet.parameters(), lr=0.001)
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [AsyncTrainingWorker(gnet, opt, global_ep, global_ep_r, res_queue, i, N_S, N_A, GAMMA, MAX_EP, SOKOBAN, ENV, UPDATE_GLOBAL) for i in range(mp.cpu_count())]
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
