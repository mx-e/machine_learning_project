import gym
import gym_sokoban
import numpy as np
import random
import torch
import copy

MODE = 'PushAndPull-Sokoban-v3'
class SokobanEnv:
    def __init__(self):
        self.env = gym.make(MODE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.action_space = self.env.action_space
        self.image_space = self.get_screen().shape

    def get_screen(self):
        no_actions = self.action_space.n
        return (copy.deepcopy(self.env.room_state).flatten()-no_actions/2)/no_actions
    
    def render(self):
        self.env.render()
        return self.get_screen()

    def step(self, action):
        self.steps_done += 1
        s_, r, done, _ =  self.env.step(action)
        if(r > 0):
            self.steps_done = 0
        if(done == True and r < 1):
            r = -1
        if(self.steps_done > 100):
            done = True
            r = -4
        return self.get_screen(), r, done, _

    def reset(self):
        self.env.reset()
        return self.get_screen()

    def close(self):
        self.env.close()
        self.steps_done = 0