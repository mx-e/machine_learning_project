
import argparse, glob, torch
import torch.nn as nn
import numpy as np

def configure_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Sokoban-small-v1', type=str, help='gym environment')
    parser.add_argument('--processes', default=56, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=100, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--save_dir', default='results', type=str, help='relative directory, in which to save models and data')
    parser.add_argument('--n_rollouts', default=5, type=int, help='no. of parallel rollouts')
    parser.add_argument('--rollout_depth', default=3, type=int, help='depth of rollouts')
    parser.add_argument('--max_on_cuda', default=20, type=int, help='depth of rollouts')
    return parser


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor / 8.
        tensor = (tensor - self.mean) / self.std
        return tensor

    def denormalize(self, tensor):
        tensor = tensor * self.std + self.mean
        tensor = tensor * 8
        return tensor


def save_modules(module_list, save_path):
    state_dicts = {}
    for key, value in module_list.items():
        state_dicts[key] = value.state_dict()
    torch.save(state_dicts, save_path)

def load_modules(module_list, save_dir):
    paths = glob.glob(save_dir + '*.tar');
    step = 0.
    if len(paths) > 0:
        ckpts = [float(s.split('_')[-2]) if not 'production' in s else 0. for s in paths]
        ix = np.argmax(ckpts);
        step = ckpts[ix]
        checkpoint = torch.load(paths[ix])
        for key, value in module_list.items():
            value.load_state_dict(checkpoint[key])
            value.train()
    print("\tno saved models") if step <= 0. else print("\tloaded models: {}".format(paths[ix]))
    return step
