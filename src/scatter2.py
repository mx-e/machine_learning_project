import matplotlib.pyplot as plt 
import numpy as np


class scatter:
	def makescatterplot(real_reward,predicted_reward,name):
		real_rew = real_reward
		pred_rew = predicted_reward
		u, c = np.unique(np.c_[real_rew,pred_rew], return_counts=True, axis=0)

		s = lambda x : (((x-x.min())/float(x.max()-x.min())+1)*8)**2

		plt.scatter(u[:,0],u[:,1],s=s(c))


		#plt.scatter(val[0,:], val[1,:], marker = '.')
		plt.xlabel('Real Reward')
		plt.ylabel('Predicted Reward')
		#plt.xlim(min(u[:,0]),max(u[:,0]))
		#plt.ylim(min(u[:,0]),max(u[:,0]))
		
		plt.xlim(min(real_rew)-0.3,max(real_rew)+0.3)
		plt.ylim(min(real_rew)-0.3,max(real_rew)+0.3)
		plt.xticks(np.arange(-1.2,1, 0.2))
		plt.yticks(np.arange(-1.2,1, 0.2))
		#plt.legend()
		#plt.show()
		plt.savefig(name)
		#plt.show()

