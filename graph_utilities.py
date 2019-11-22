import numpy as np 
import random


# Bayesian Graph with variable names topologically sorted
class graph:
	def __init__(self, parents, distribution):
		self.variables = np.arange(len(parents))
		self.parents = parents
		self.distribution = distribution
		random.seed(4)

	def intervention(self, assignment):
		ret = {}
		for v in self.variables:
			if v in assignment:
				ret[v] = assignment[v]
				continue
			x = self.distribution[v]
			for p in self.parents[v]:
				x = x[assignment[p]]
			if random.random() < x[0]:
				ret[v] = 0
			else:
				ret[v] = 1
		return ret