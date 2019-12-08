# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
from sokoban_env import SokobanEnv
import torch, os, gym, time, glob, argparse, sys
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
    parser.add_argument('--rnn_steps', default=100, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
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
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256,256)
        self.gru = nn.GRUCell(256, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.linear1(inputs))
        x = F.elu(self.linear2(x))
        hx = self.gru(x, (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

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


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    #print(actions.clone().detach().view(-1, 1))
    #print(logps)
    logpys = logps.gather(1, actions.clone().detach().view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum()  # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


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

    while info['frames'][0] <= 5e7 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict())  # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], []  # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((prepro(state), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()

            #state = torch.tensor(state);
            epr += reward
            reward = np.clip(reward, -1, 10)  # reward
            done = done or episode_length >= 1e4  # don't playing one ep for too long

            info['frames'].add_(1);
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:  # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
                torch.save(shared_model.state_dict(), args.save_dir + 'model.{:.0f}.tar'.format(num_frames / 1e6))

            if done:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                         .format(elapsed, info['episodes'].item(), num_frames / 1e6,
                                 info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = env.reset()

            values.append(value);
            logps.append(logp);
            actions.append(action);
            rewards.append(reward)

        next_value = torch.zeros(1, 1) if done else model((prepro(state), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad();
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!"  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower())  # keep the directory structure simple
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