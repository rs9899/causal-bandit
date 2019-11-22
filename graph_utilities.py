import numpy as np 
import random


# Bayesian Graph with variable names topologically sorted
class graph:
	def __init__(self, parents, distribution):
		self.variables = np.arange(len(parents))
		self.parents = parents
		self.distribution = distribution
		random.seed(40)

	def intervention(self, assignment):
		returnDict = {}
		for v in self.variables:
			if v in assignment:
				returnDict[v] = assignment[v]
				continue
			x = self.distribution[v]
			for p in self.parents[v]:
				x = x[returnDict[p]]
			if random.random() < x:
				returnDict[v] = 1
			else:
				returnDict[v] = 0
		return returnDict
