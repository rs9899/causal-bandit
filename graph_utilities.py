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
		for v in self.variables:
			if v in assignment:
				continue
			x = self.distribution[v]
			for p in self.parents[v]:
				x = x[assignment[p]]
			if random.random() < x[0]:
				assignment[v] = 0
			else:
				assignment[v] = 1
		return assignment