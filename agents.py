"""
Class descriptions for all agents.
"""

import numpy as np

class Agent(object):
	def __init__(self, G, A):
		self.graph = G
		self.actions = A

	def _step(self):
		raise NotImplementedError

	def run(self, horizon=100):
		raise NotImplementedError

class UCBAgent(Agent):
	def __init__(self, G, A):
		super(UCBAgent, self).__init__(G, A)






class OCTSAgent(Agent):
	def __init__(self, G, A):
		super(OCTSAgent, self).__init__(G, A)
		self.numAction = len(G.variables) - 1
		self.numPartition = 2 ** (self.numAction)
		self.beta = np.zeros((self.numPartition,2), dtype=int) + 1
		self.dirch = np.zeros((self.numPartition,2 * self.numAction), dtype=int) + 1

	def _step(self):
		succesChance = np.zeros(2 * self.numAction)
		for a in range(2 * self.numAction):
			# v   = a // 2
			# val = a %  2
			partionProb = np.random.dirichlet(self.dirch[:,a]).reshape((self.numPartition,1))
			sampl = np.random.beta(self.beta[:,0],self.beta[:,1]).reshape((1 , self.numPartition))
			succesChance[a] = np.asscalar(sampl @ partionProb)
		
		best = np.argmax(succesChance)
		v = best // 2
		val = best %  2
		d = dict()
		d[v] = val

		d = G.intervention(d) 

		r = d[self.numAction]
		z = 0
		for i in range(self.numAction):
			z = z + ( 2**i * d[i])

		self.dirch[z,best] += 1

		if r == 1:
			self.beta[z,0] += 1
		else:
			self.beta[z,1] += 1