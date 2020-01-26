#!/usr/bin/env python3
#$ -V
#$ -cwd
#$ -binding linear:20
#$ -l cuda=4

from __future__ import print_function
import torch, os, gym, time, sys
import numpy as np

from scipy.signal import lfilter
import torch.nn.functional as F
import torch.multiprocessing as mp
import cProfile

sys.path.append('./src')
sys.path.append('./src/modules')
sys.path.append('./env_model')
from sokoban_env import SokobanEnv
from shared_adam import SharedAdam
from conv2d_module import Conv2d_Module
from linear_module import Linear_Module
from rollout_lstm_module import Rollout_LSTM_Module
from environment_module import Env_Module
from rollout_unit import RolloutUnit
from utils import configure_parser, save_modules, load_modules, update_shared_info, update_log, printlog
from policy_output_module import Policy_Output_Module

os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = configure_parser()
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner

def cost_func(args, values, logps, actions, rewards, copy_policy_logps):

    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    # print(actions.clone().detach().view(-1, 1))
    # print(logps)
    logpys = logps.gather(1, actions.clone().detach().view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()
    cross_entropy_loss = (-copy_policy_logps * torch.exp(logps)).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum()  # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss - 0.01 * cross_entropy_loss


class I2A_PipeLine:
    def __init__(self, modules, env_module, args, gpu, cpu):
        self.args = args
        self.model_free_conv = modules['model_free_conv']
        self.rollout_conv = modules['rollout_conv']
        self.linear_output = modules['linear_output']
        self.rollout_lstm = modules['rollout_lstm']
        self.policy_output = modules['policy_output']
        self.rollout_unit = RolloutUnit(args, self.model_free_conv, self.rollout_conv, self.rollout_lstm, env_module, self.policy_output, gpu, cpu)

    def pipe(self, input):
        rollout_encoding = self.rollout_unit.make_rollout_encoding(input)
        conv_output = self.model_free_conv(input).flatten()
        output_layer_input = torch.cat((rollout_encoding, conv_output)).unsqueeze(0)
        logp, value = (self.linear_output(output_layer_input))
        copy_policy_logp = self.policy_output(self.model_free_conv(input))
        return (logp, value, copy_policy_logp)

# for profiling purposes
# def test_train(shared_modules, shared_optim, rank, args, info):
#    cProfile.runctx('train(shared_modules, shared_optim, rank, args, info)', globals(), locals(), 'prof%d.prof' % rank)

def train(shared_modules, shared_optim, rank, args, info):
    cuda = args.cuda and rank <= args.max_on_cuda
    gpu = torch.device(f'cuda:{rank % args.cuda_count}') if cuda else None
    cpu = torch.device('cpu')
    if(cuda):
        print(f"Worker {rank} on GPU No. {rank%args.cuda_count} ({torch.cuda.get_device_name(rank%args.cuda_count)}, {torch.cuda.get_device_capability(rank%args.cuda_count)}) ")
    else:
        print(f"Worker {rank} on CPU")
    env = SokobanEnv()  # make a local (unshared) environment
    env.seed()
    torch.manual_seed(args.seed + rank)  # seed everything
    env_module = Env_Module(input_size=args.input_size, num_actions=args.num_actions)#.to(gpu if args.cuda else cpu)
    env_module.load_state_dict(torch.load(F"./env_model/envs/{args.env.lower()}/production.tar"))
    env_module.eval()
    modules = {
        'rollout_conv': Conv2d_Module(is_sokoban=True).to(gpu if cuda else cpu),
        'model_free_conv': Conv2d_Module(is_sokoban=True),
        'linear_output': Linear_Module(args.output_module_input_size, args.num_actions, is_sokoban=True),
        'rollout_lstm': Rollout_LSTM_Module(input_size=args.rollout_lstm_input_size, is_sokoban=True).to(gpu if cuda else cpu),
        'policy_output': Policy_Output_Module(input_size = args.conv_output_size, num_action = args.num_actions)
    }
    model_pipeline = I2A_PipeLine(modules, env_module, args, gpu, cpu)
    state = env.reset()  # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, no_boxes, done = 0, 0, 0, 0, True  # bookkeeping

    while info['frames'][0] <= 5e7 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        for module, shared_module in zip(modules.values(), shared_modules.values()):
            if next(module.parameters()).is_cuda:
                with torch.cuda.device(rank%args.cuda_count):
                    module.load_state_dict(shared_module.state_dict())
            else:
                module.load_state_dict(shared_module.state_dict())

        values, logps, actions, rewards, copy_policy_logps = [], [], [], [], []  # save values for computing gradientss

        while (True):
            episode_length += 1
            logp, value, copy_policy_logp = model_pipeline.pipe(state.unsqueeze(0))
            logp = F.log_softmax(logp, dim=-1)
            copy_policy_logp = F.log_softmax(copy_policy_logp, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()
            if reward >= 0.9:
                no_boxes+=1
            if reward <= -0.9:
                no_boxes-=1
            epr += reward
            reward = np.clip(reward, -1, 10)  # reward

            info['frames'].add_(1);
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:  # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
                save_modules(shared_modules, args.save_dir + 'model_{:.2f}_.tar'.format(num_frames / 1e6))

            if done:  # update shared data
                update_shared_info(args, eploss, epr, info, no_boxes)

            if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
                update_log(args, info, num_frames, start_time)
                last_disp_time = time.time()

            if done:  # maybe print info.
                episode_length, epr, eploss, no_boxes = 0, 0, 0, 0
                state = env.reset()
                break

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)
            copy_policy_logps.append(copy_policy_logp)

        next_value = torch.zeros(1, 1) if done else model_pipeline.pipe(state.unsqueeze(0))[1]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards), torch.cat(copy_policy_logps))
        eploss += loss.item()
        loss.backward()

        for module in modules.values():
            torch.nn.utils.clip_grad_norm_(module.parameters(), 40)

        shared_optim.zero_grad()
        for module, shared_module in zip(modules.values(), shared_modules.values()):
            if next(module.parameters()).is_cuda:
                for param, shared_param in zip(module.parameters(), shared_module.parameters()):
                    if shared_param.grad is None: shared_param._grad = param.grad.cpu()  # if gpu, convert
            else:
                for param, shared_param in zip(module.parameters(), shared_module.parameters()):
                    if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optim.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    args = get_args()
    args.cuda_count = torch.cuda.device_count()
    args.cuda = args.cuda_count > 0
    if(args.cuda):
        torch.cuda.manual_seed(args.seed)
    args.save_dir = f'{args.save_dir}/{args.env.lower()}/'  # keep the directory structure simple
    if args.render:  args.processes = 1; args.test = True  # render mode -> test mode w one process
    if args.test:  args.lr = 0  # don't train in render mode
    env = SokobanEnv()
    args.num_actions = env.action_space
    args.input_size = env.observation_space.size()

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # make dir to save models etc.
    test_net = Conv2d_Module(is_sokoban=True)
    test_img = env.render().unsqueeze(0)
    args.conv_output_size = test_net(test_img).view(-1).size()[0]
    args.rollout_lstm_input_size = args.conv_output_size + list(env.observation_space.flatten().size())[0]
    args.output_module_input_size = 1024 + args.conv_output_size

    torch.manual_seed(args.seed)
    shared_modules = {
        'rollout_conv': Conv2d_Module(is_sokoban=True).share_memory(),
        'model_free_conv': Conv2d_Module(is_sokoban=True).share_memory(),
        'linear_output': Linear_Module(args.output_module_input_size, args.num_actions, is_sokoban=True).share_memory(),
        'rollout_lstm': Rollout_LSTM_Module(input_size = args.rollout_lstm_input_size, is_sokoban=True).share_memory(),
        'policy_output': Policy_Output_Module(input_size = args.conv_output_size, num_action = args.num_actions).share_memory()
    }

    parameters = set()
    for module in shared_modules.values():
        parameters |= set(module.parameters())

    shared_optim = SharedAdam(parameters, lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames', 'one_box', 'two_boxes', 'three_boxes']}
    info['frames'] += int(load_modules(shared_modules, args.save_dir) * 1e6)
    if int(info['frames'].item()) == 0: printlog(args, '', end='', mode='w')  # clear log file

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_modules, shared_optim, rank, args, info))
        p.start();
        processes.append(p)
    for p in processes: p.join()
