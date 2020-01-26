''''This Method is for translating the gamestate values to rgb-values and saving an Image
 
 - picturename has to be unique!
 - colors will be mapped between blue and red
 	- blue : walls
 	- orange: player (sokoban)
 	- green: boxes and fields
 '''




from sokoban_env import SokobanEnv
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torchvision
from torchvision import transforms


def prediction_screening(environment_state, picturename):


	gamestate = environment_state
	
	gamestate = transforms.ToPILImage(mode = 'RGB')(gamestate)
	rgb_array = gamestate
	
	rgb_array=rgb_array.resize((400,400))
	rgb_array.save(picturename,"PNG")
