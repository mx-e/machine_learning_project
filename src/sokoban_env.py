import gym
import gym_sokoban
import numpy as np
import random
import torch
import copy

MODE = 'Sokoban-small-v1'
ROOM_ENCODING_SIZE = 6
class SokobanEnv:
    def __init__(self):
        self.env = gym.make(MODE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.action_space = self.env.action_space
        self.observation_space = self.get_screen()
        self.room_state = self.env.room_state


    def get_screen(self):
        no_actions = self.action_space.n
        return torch.Tensor((copy.deepcopy(self.env.room_state))/ROOM_ENCODING_SIZE).unsqueeze(0)
    
    def render(self):
        self.env.render()
        return self.get_screen()

    def step(self, action):
        self.steps_done += 1
        s_, r, done, _ =  self.env.step(action)
        if(self.steps_done > 120):
            done = True
        if(done == True and r < 1):
            r = -0.5
        return self.get_screen(), r, done, _

    def reset(self):
        self.env.reset()
        self.steps_done = 0
        return self.get_screen()

    def close(self):
        self.env.close()
        self.steps_done = 0

    def get_room_state(self):
        return self.room_state

    def seed(self):
        return self.env.seed()