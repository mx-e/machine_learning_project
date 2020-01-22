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
        self.action_space = 4 #reducing action space (push and move in a direction is evaluated), no no-op
        self.observation_space = self.getEnvState()
        self.room_state = self.env.room_state

    def getEnvState(self):
        return torch.tensor(self.env.room_state).unsqueeze(0)

    def get_screen(self):
        return torch.from_numpy(self.env.render('tiny_rgb_array')).permute(2,0,1).float()/255
    
    def render(self):
        self.env.render('tiny_rgb_array')
        return self.get_screen()

    def step(self, action):
        self.steps_done += 1
        r, done =  self.eval_step(action)
        if(self.steps_done > 120):
            done = True
            r = -0.1
        return self.get_screen(), r, done, None

    def eval_step(self, action):
        location = np.argwhere(self.env.room_state == 5)
        x = location[0, 1]
        y = location[0, 0]
        print(x,y,action)
        if(action == 0):
            obstacle = self.env.room_state[y-1,x]
        elif(action == 1):
            obstacle = self.env.room_state[y+1,x]
        elif(action == 2):
            obstacle = self.env.room_state[y,x-1]
        else:
            obstacle = self.env.room_state[y,x+1]
        if(obstacle != 3 and obstacle !=4):
            _, r, done, _ = self.env.step(action + 5) # move, if there is no box or box on a target
        else:
            _, r, done, _ = self.env.step(action + 1) # otherwise, push box
        return (r, done)

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