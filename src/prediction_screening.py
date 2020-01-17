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


def prediction_screening(environment_state, picturename):

	#print(type(testenvironment))
	gamestate = environment_state
	#gamestate = environment_state.get_screen() 
	#print(gamestate.size())
	#print(gamestate)


	gamestate=gamestate.numpy()

	gamestate_reshaped=gamestate.reshape((7,7,1))
	rgb_array = gamestate.reshape((7,7,1))

	rgb_extension = np.zeros((7,7,2))
	#gamestate_reshaped = np.append(gamestate_reshaped,rgb_extension,axis=-1)
	rgb_array = np.append(rgb_array,rgb_extension,axis=-1)


	def rgb(minimum, maximum, value):
	    minimum, maximum = float(minimum), float(maximum)
	    ratio = 2 * (value-minimum) / (maximum - minimum)
	    b = int(max(0, 255*(1 - ratio)))
	    r = int(max(0, 255*(ratio - 1)))
	    g = 255 - b - r
	    return r, g, b

	smallest = 0
	biggest = 1

	for i in range(len(gamestate_reshaped)):
		for j in range(len(gamestate_reshaped)):
			rgb_array[i,j] = rgb(smallest,biggest, gamestate_reshaped[i,j])
			#print(rgb_array[i,j])
			#print(gamestate_reshaped[i,j])
	#print(rgb_array)
	#print(rgb_array.shape)
	rgb_array = np.array(rgb_array,dtype=np.uint8) 

	img = Image.fromarray(rgb_array,'RGB')
	img.save(picturename)




'''
environment= SokobanEnv()
prediction_screening(environment.get_screen(),"pic0.png") 
nxt_state,r,done,_= environment.step(1)
print("hallos")
print(environment.get_screen())
prediction_screening(nxt_state,"pic01.png")
nxt_state,r,done,_= environment.step(2)
print("hallos")
print(environment.get_screen())
prediction_screening(nxt_state,"pic02.png")
print("sack")
print(nxt_state)
''


for i in range(3):
	action = i
	name = "picture " + str(i) +".png"

	prediction_screening(environment,name)
	environment= environment.step(action)
'''


