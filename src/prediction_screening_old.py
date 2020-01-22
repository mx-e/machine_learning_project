''''This Method is for translating the gamestate values to rgb-values and saving an Image as PNG
takes 3x7x7 torch tensor!
 
 
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

	#img = img.resize( (400, 400))
	
	#img.save(picturename, "PNG")

	
	





