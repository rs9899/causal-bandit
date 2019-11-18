import numpy as np 
import random


# Bayesian Graph with variable names topologically sorted
class graph:
	def __init__(self, parents, distribution):
		self.variables = np.arange(len(parents))
		x = [[]]*len(parents)
		self.parents = parents
		self.distribution = distribution
		for i in range(len(parents)):
			for j in range(parents[i]):
				x[parents[i,j]].append(i)
		self.children = np.asarray(x)
		random.seed(4)

	def intervention(self, assignment):
		# unsampled_parents = {}
		# for i in self.variables:
		# 	unsampled_parents[i] = len(parents[i])
		# for i in assignment:
		# 	for j in self.children[i]:
		# 		unsampled_parents[j] -= 1

		# variables_to_sample = []
		# for i in self.variables:
		# 	if i not in assignment and unsampled_parents[i]==0:
		# 		variables_to_sample.append(i)

		# n = len(variables_to_sample)
		# i = 0
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
			# for c in self.children[v]:
			# 	unsampled_parents[c] -= 1
			# 	if unsampled_parents[c] == 0 and c not in assignment:
			# 		variables_to_sample.append(c)
			# 		n += 1
			# i+=1
		return assignment