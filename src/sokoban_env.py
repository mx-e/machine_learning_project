import gym
import gym_sokoban
import numpy as np
import random
import math
import torch
import copy
import torchvision.transforms as T
from PIL import Image

class SokobanEnv:
    def __init__(self, env_mode = 'Sokoban-small-v1', eps_end = 0.05, eps_start = 0.95, eps_decay = 20000):
        self.env = gym.make(env_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = self.env.action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        #self.room_state = copy.deepcopy(self.env.room_state)
        #self.box_mapping = copy.deepcopy(self.env.box_mapping)
        #self.room_fixed = copy.deepcopy(self.env.room_fixed)

    def get_screen(self):
        screen =  np.ascontiguousarray(self.get_board(), dtype=np.float32).flatten()
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return screen.to(self.device)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        return self.get_screen()
        #self.env.room_state = copy.deepcopy(self.room_state)
        #self.env.room_fixed = copy.deepcopy(self.room_fixed)
        #self.env.box_mapping = copy.deepcopy(self.box_mapping)
        #self.env.player_position = np.argwhere(self.env.room_state == 5)[0]
        #self.env.num_env_steps = 0
        #self.env.reward_last = 0
        #self.env.boxes_on_target = 0

    def get_board(self):
        return copy.deepcopy(self.env.room_state)

    def get_no_of_solved_boxes(self):
        return len(np.argwhere(self.env.room_state == 3))

    def close(self):
        self.env.close()
        self.steps_done = 0
