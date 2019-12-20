# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys, random
import numpy as np
from scipy.signal import lfilter
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym_sokoban

os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-ram-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=8, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=10, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda state: torch.tensor((state.astype(np.float32)/255)).unsqueeze(0)


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end);
    f = open(args.save_dir + 'log.txt', mode);
    f.write(s + '\n');
    f.close()


class NNPolicy(nn.Module):  # an actor-critic neural network
    def __init__(self, input_size, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.action_space = num_actions
        self.input_size = input_size
        self.conv1 = torch.nn.Conv1d(in_channels=self.action_space + 1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.deconf1 = torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.val_conv1 = self._make_stage(32, 32)
        self.val_conv2 = self._make_stage(32, 1)
        self.val_linear = nn.Linear(5, 5)
        self.val_output = nn.Linear(5, 1)


    def forward(self, inputs, train=True, hard=False):
        state, action = inputs
        one_hot_action_encoding = np.zeros(self.action_space)
        one_hot_action_encoding[action] = 1
        one_hot_action_encoding = one_hot_action_encoding.reshape(4,1)
        one_hot_action_encoding = np.tile(one_hot_action_encoding,(self.input_size)).astype(np.float32)

        model_input = torch.cat([state, torch.tensor(one_hot_action_encoding)]).unsqueeze(0)
        x = F.relu(self.conv1(model_input))
        x = F.relu(self.conv2(x))
        predicted_state = self.deconf1(x)

        x = self.val_conv1(x)
        x = self.val_conv2(x)
        print(x.size())
        x = F.softmax(self.val_linear(x), dim=1)
        predicted_value = self.val_output(x)

        return predicted_state, predicted_value

    def _make_stage(self, in_channels, out_channels):
        stage = nn.Sequential()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            padding=1,
                )

        stage.add_module('conv', conv)
        stage.add_module('relu', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=1))
        return stage

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar');
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts);
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class SharedAdam(torch.optim.Adam):  # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
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


def cost_func(args, states, predicted_states, rewards, predicted_rewards):
    return (states-predicted_states).pow(2).sum()

def train(shared_model, shared_optimizer, rank, args, info):
    if(args.sokoban):
        env = SokobanEnv()  # make a local (unshared) environment
    else: 
        env = gym.make(args.env)
    env.seed();
    torch.manual_seed(args.seed + rank)  # seed everything
    model = NNPolicy(input_size=args.input_size, memsize=args.hidden, num_actions=args.num_actions)  # a local/unshared model
    state = env.reset() # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True  # bookkeeping

    while info['frames'][0] <= 4e7 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict())  # sync with shared model
        states, predicted_states, rewards, predicted_rewards = [], [], [], []  # save values for computing gradients

        for step in range(args.rnn_steps):
            episode_length += 1
            action = random.sample(range(args.num_actions), 1)[0]
            predicted_state, predicted_reward = model((prepro(state), action))
    
            state, reward, done, _ = env.step(action)
            if args.render: env.render()

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

            states.append(prepro(state))
            predicted_states.append(predicted_state)
            rewards.append(reward)
            predicted_rewards.append(predicted_reward)

        loss = cost_func(args, torch.cat(states), torch.cat(predicted_states), torch.cat(rewards), np.asarray(predicted_rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad();
        loss.backward()

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!"  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = 'envs/{}/'.format(args.env.lower())  # keep the directory structure simple
    if args.render:  args.processes = 1; args.test = True  # render mode -> test mode w one process
    if args.test:  args.lr = 0  # don't train in render mode
    if(args.env == 'Sokoban-small-v1'): 
        args.sokoban = True
        None
    else:
        args.num_actions = gym.make(args.env).action_space.n  # get the action space of this game
        args.input_size = gym.make(args.env).reset().size
        args.sokoban = False

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(input_size=args.input_size, memsize=args.hidden, num_actions=args.num_actions).share_memory()
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