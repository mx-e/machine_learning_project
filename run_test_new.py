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
from performance_insights import *

sys.path.append('./src')
from prediction_screening import prediction_screening
from scatter2 import scatter
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
import random

os.environ['OMP_NUM_THREADS'] = '1'

class run_test:

	env1 = SokobanEnv() #initialize environment
	state = env1.get_screen()

	real_pic_name1 = "game_state"
	pred_pic_name1 = "predicted_game_state"
	path_dict_env_module1 = "/Users/heiko.langer/GIT/machine_learning_project/env_model/envs/sokoban-small-v1/model.2.tar"
	possible_actions = [0,1,2,3]
	action = random.choices(possible_actions,k=100)
	#print(action)
	#reward_comparison = np.empty(2,1)
	reward_array_real = np.empty([1,1])
	reward_array_model = np.empty([1,1])
	counter = 1
	name = "scatter_plot.png"
	for i in action:
	    counter+= 1
	    i = int(i)
	    real_nxt_state1, reward_real1 ,model_predictions_norm1 , reward_model1 = performance_insights.prediction_comparison(path_dict_env_module1,env1,state,i)
	    
	    reward_model1 = float(reward_model1.data)
	    model_predictions_norm1 = model_predictions_norm1[0,:,:,:]
	    #print(str(reward_real1)+" predicted "+ str(reward_model1))
	    reward_array_real = np.append(reward_array_real,reward_real1)
	    reward_array_model = np.append(reward_array_model,reward_model1)


	    real_pic_name = "real" + str(counter).zfill(3) 
	    performance_insights.savepics(real_nxt_state1, real_pic_name)
	    model_pic_name = "model" + str(counter).zfill(3)  
	    performance_insights.savepics(model_predictions_norm1, model_pic_name)

	    #img_sidebyside_array = np.concatenate((real_nxt_state1,model_predictions_norm1), axis = 1)
	    #print(img_sidebyside_array.shape)
	    #print(img_sidebyside_array)


	    #img_sidebyside=Image.fromarray(img_sidebyside_array,'RGB')
	    #img_sidebyside = img_sidebyside.resize( (800, 400))
	    #img_sidebyside.save('skks.png', "PNG")

		

	    state = real_nxt_state1

	#print(reward_array_real)
	#print(reward_array_model)


	reward_array_both = np.empty((len(action),1))
	#print(reward_array_both.ndim)
	reward_array_both = np.vstack((reward_array_real,reward_array_model))#, axis = 1)
	#print(reward_array_both)
	#scatter.makescatterplot(reward_array_both,name)

	scatter.makescatterplot(reward_array_real,reward_array_model,name)

