import copy
import gym
import gym_sokoban
import torch.multiprocessing as mp

from utils import v_wrap, push_and_pull, record
from net import Net
from sokoban_env import SokobanEnv

class AsyncTrainingWorker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, input_dim, output_dim, gamma, max_ep, sokoban, env, update_global):
        super(AsyncTrainingWorker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(input_dim, output_dim)           # local network
        if sokoban:
            self.env = SokobanEnv()
        else:
            self.env = gym.make(env).unwrapped
        self.gamma = gamma
        self.max_ep = max_ep
        self.update_global = update_global

    def run(self):
        total_step = 1
        while self.g_ep.value < self.max_ep:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)
        