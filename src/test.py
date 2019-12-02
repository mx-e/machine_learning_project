from sokoban_env import SokobanEnv
import gym
import gym_sokoban
import numpy as np
import random
import torch
import copy

env = SokobanEnv()
print(env.get_room_state())
