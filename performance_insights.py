from __future__ import print_function

from PIL import Image
import torch, os, gym, time, sys
import matplotlib
import matplotlib.cm as cm
import numpy as np
from scipy.signal import lfilter
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
sys.path.append('./src')
from prediction_screening import prediction_screening
from scatter import scatter
sys.path.append('./src/modules')
sys.path.append('./env_model')
from sokoban_env import SokobanEnv
from shared_adam import SharedAdam
from conv2d_module import Conv2d_Module
from linear_module import Linear_Module
from rollout_lstm_module import Rollout_LSTM_Module
from environment_module import Env_Module
from rollout_unit import RolloutUnit
from utils import configure_parser
from policy_output_module import Policy_Output_Module

os.environ['OMP_NUM_THREADS'] = '1'
'''
path_dict_env - give path of learned weights of the environmental.
 Must be ".tar" and handed over as string "/Users/heiko.langer/TUB/ProjML/machine_learning_project-i2a-sokoban-v1/env_model/envs/sokoban-small-v1/production.tar"
environment - give game environment: here SokobanENV()
state - current state has to be tensor of shape (1,7,7)
action - single integer for sokoban between [0:4]
'''
class performance_insights:
    def prediction_comparison (path_dict_env_module,environment,state, action):
        sigmoid= nn.Sigmoid()
        input_size = environment.observation_space.size()
        num_actions = environment.action_space
        reward_real = 0
        reward_model = 0


        env = environment
        action1 = action

        env_module = Env_Module(input_size=input_size, num_actions=num_actions)
        env_module.load_state_dict(torch.load(path_dict_env_module))
        env_module.eval()

        real_state_0 = state #initialize same gamestate to begin with
        model_state0 = state
        model_nxt_state = state

        real_nxt_state,reward_real,done,_= env.step(action1) #do step with action

        model_nxt_state, reward_model = env_module.forward((model_nxt_state, action1)) #self.env_module((model_nxt_state.squeeze(0), action))
        model_nxt_state = sigmoid(model_nxt_state)
        model_predictions_norm = model_nxt_state.data
        
        return real_nxt_state, reward_real, model_predictions_norm, reward_model


    def savepics(state, pic_name):
        pic_name = pic_name +".png"
        prediction_screening(state, pic_name) #print origin state

        
        




    
