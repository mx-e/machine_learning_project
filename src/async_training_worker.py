import torch.multiprocessing as mp
from .nn import NN
from .sokoban_env import SokobanEnv
import torch
import numpy as np

MAX_EP = 10000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99

class AsyncTrainingWorker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_ep_r, res_queue, name, input_size, output_size):
        super(AsyncTrainingWorker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_episode, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = NN(input_size, output_size)           # local network
        self.env = SokobanEnv('Sokoban-small-v1', eps_end = 0.05, eps_start = 0.95, eps_decay = 20000)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            self.env.reset()
            s = self.env.get_screen()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w0':
                    self.env.get_screen()
                a = self.local_net.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                s_ = self.env.get_screen()
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    self.push_and_pull(self.optimizer, self.local_net, self.global_net, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        self.record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)

    def push_and_pull(self, opt, lnet, gnet, done, s_, bs, ba, br, gamma):
        ba = np.ndarray(ba)
        if done:
            v_s_ = 0.  # terminal
        else:
            v_s_ = lnet.forward(s_)[0].data.numpy()[0]

        buffer_v_target = []
        for r in br[::-1]:  # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = lnet.loss_func(
            self.v_wrap(np.vstack(bs)),
            self.v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else self.v_wrap(np.vstack(ba)),
            self.v_wrap(np.array(buffer_v_target)[:, None]))

        # calculate local gradients and push local parameters to global
        opt.zero_grad()
        loss.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp.grad
        opt.step()

        # pull global parameters
        lnet.load_state_dict(gnet.state_dict())

    def record(self, global_ep, global_ep_r, ep_r, res_queue, name):
        with global_ep.get_lock():
            global_ep.value += 1
        with global_ep_r.get_lock():
            if global_ep_r.value == 0.:
                global_ep_r.value = ep_r
            else:
                global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
        res_queue.put(global_ep_r.value)
        print(
            name,
            "Ep:", global_ep.value,
            "| Ep_r: %.0f" % global_ep_r.value,
        )