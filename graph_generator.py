import numpy as np
from graph_utilities import graph

class graph_samples:
	def __init__(self, num_variables):
		self.num_variables = num_variables
		np.random.seed(7)

	def get_distribution(self, variables, parents):
		distribution = {}
		for i in variables:
			if len(parents[i])==0:
				distribution[i] = np.random.rand(1)[0]
				continue
			distribution[i] = np.random.rand(2**len(parents[i]))
			l = [2]*len(parents[i])
			distribution[i] = np.reshape(distribution[i], tuple(l))
		return distribution

	def linear_graph(self):
		variables = np.arange(self.num_variables)
		parents = [[]]*self.num_variables
		for i in range(1,num_variables):
			parents[i].append(i-1)
		parents = np.asarray(parents)
		return graph(parents, get_distribution(variables, parents))

	def disjoint_graph(self):
		variables = np.arange(self.num_variables)
		parents = [[]]*self.num_variables
		for i in range(self.num_variables-1):
			parents[self.num_variables-1].append(i)
		parents = np.asarray(parents)
		return graph(parents, get_distribution(variables, parents))

	def random_graph(self):
		variables = np.arange(self.num_variables)
		parents = [[]]*self.num_variables
		for i in range(1,num_variables):
			x = np.random.randint(2, size=i)
			for j in range(i):
				if x[j]==1:
					parents[i].append(j)
		parents = np.asarray(parents)
		return graph(parents, get_distribution(variables, parents))