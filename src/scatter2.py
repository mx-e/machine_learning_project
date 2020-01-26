import matplotlib.pyplot as plt 
import numpy as np


class scatter:
	def makescatterplot(real_reward,predicted_reward,name):
		real_rew = real_reward
		pred_rew = predicted_reward
		u, c = np.unique(np.c_[real_rew,pred_rew], return_counts=True, axis=0)

		s = lambda x : (((x-x.min())/float(x.max()-x.min())+1)*8)**2

		plt.scatter(u[:,0],u[:,1],s=s(c))


		
		plt.xlabel('Real Reward')
		plt.ylabel('Predicted Reward')
		
		
		
		plt.xlim(min(real_rew)-0.3,max(real_rew)+0.3)
		plt.ylim(min(real_rew)-0.3,max(real_rew)+0.3)
		plt.xticks(np.arange(-1.2,1, 0.2))
		plt.yticks(np.arange(-1.2,1, 0.2))
		
		
		plt.savefig(name)
		

