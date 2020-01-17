import matplotlib.pyplot as plt 
import numpy as np


class scatter:
	def makescatterplot(values,name):
		val=values
		plt.scatter(val[0,:], val[1,:], marker = '.')
		#plt.show()
		plt.savefig(name)

