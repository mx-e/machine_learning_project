# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, time, argparse, sys, random
import numpy as np
from scipy.signal import lfilter
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable

sys.path.append('../src')
sys.path.append('./')
from sokoban_env import SokobanEnv
from environment_module import Env_Module

os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Sokoban-small-v1', type=str, help='default sokoban version')
    parser.add_argument('--processes', default=8, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=1, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda state: torch.tensor((state.astype(np.float32) / 255)).unsqueeze(0)


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end);
    f = open(args.save_dir + 'log.txt', mode);
    f.write(s + '\n');
    f.close()

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class SharedAdam(torch.optim.Adam):  # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1  # a "step += 1"  comes later
            super.step(closure)

def train(shared_model, shared_optimizer, rank, args, info):
    env = SokobanEnv()  # make a local (unshared) environment
    torch.manual_seed(args.seed + rank)  # seed everything
    model = Env_Module(input_size=args.input_size, num_actions=args.num_actions)  # a local/unshared model
    state = env.reset()  # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True  # bookkeeping

    while info['frames'][0] <= 1e8 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict())  # sync with shared model
        episode_length += 1
        action = torch.tensor(random.sample(range(args.num_actions), 1)[0])
        predicted_state, predicted_reward = model((state, action))

        state, reward, done, _ = env.step(action.item())
        if args.render: env.render()

        predicted_state = predicted_state.squeeze(0)
        epr += reward
        reward = torch.tensor(reward).unsqueeze(0)
        done = done or episode_length >= 1e4  # don't playing one ep for too long

        info['frames'].add_(1);
        num_frames = int(info['frames'].item())

        if num_frames % 2e6 == 0:  # save every 2M frames
            printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
            torch.save(shared_model.state_dict(), args.save_dir + 'model.{:.0f}.tar'.format(num_frames / 1e6))

        if done:  # update shared data
            info['episodes'] += 1
            interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
            info['run_loss'].mul_(1 - interp).add_(interp * eploss)

        if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
            elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
            printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, run loss {:.2f}'
                     .format(elapsed, info['episodes'].item(), num_frames / 1e6, info['run_loss'].item()))
            last_disp_time = time.time()

        if done:  # maybe print info.
            episode_length, epr, eploss = 0, 0, 0
            state = env.reset()

        criterion = nn.BCELoss()
        loss_value = criterion(predicted_state.view(1,-1), state.view(1,-1))
        loss_value += (reward - predicted_reward).pow(2).squeeze()
        eploss += loss_value.item()

        loss_value.backward()
        shared_optimizer.zero_grad()
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope

    args = get_args()
    args.save_dir = 'envs/{}/'.format(args.env.lower())  # keep the directory structure simple
    if args.render:  args.processes = 1; args.test = True  # render mode -> test mode w one process
    if args.test:  args.lr = 0  # don't train in render mode
    env = SokobanEnv()
    args.num_actions = env.action_space
    args.input_size = env.observation_space.size()

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = Env_Module(input_size=args.input_size, num_actions=args.num_actions).share_memory()
    
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args, '', end='', mode='w')  # clear log file

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start();
        processes.append(p)
    for p in processes: p.join()