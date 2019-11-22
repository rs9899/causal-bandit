"""
Class for generating random graphs of varying topology.
"""

import numpy as np
from graph_utilities import Graph

class GraphSampler(object):

	@staticmethod
	def get_distribution(variables, parents):
		distribution = {}
		for i in variables:
			if len(parents[i])==0:
				distribution[i] = np.random.rand(1)[0]
				continue
			distribution[i] = np.random.rand(2**len(parents[i]))
			l = [2]*len(parents[i])
			distribution[i] = np.reshape(distribution[i], tuple(l))
		return distribution

	@staticmethod
	def linear_graph(num_variables):
		variables = np.arange(num_variables)
		parents = [[] for _ in range(num_variables)]
		for i in range(1, num_variables):
			parents[i].append(i-1)
		parents = np.asarray(parents)
		return Graph(parents, GraphSampler.get_distribution(variables, parents))

	@staticmethod
	def disjoint_graph(num_variables):
		variables = np.arange(num_variables)
		parents = [[] for _ in range(num_variables)]
		for i in range(num_variables-1):
			parents[num_variables-1].append(i)
		parents = np.asarray(parents)
		return Graph(parents, GraphSampler.get_distribution(variables, parents))

	@staticmethod
	def random_graph(num_variables):
		variables = np.arange(num_variables)
		parents = [[] for _ in range(num_variables)]
		for i in range(1,num_variables):
			x = np.random.randint(2, size=i)
			for j in range(i):
				if x[j]==1:
					parents[i].append(j)
		parents = np.asarray(parents)
		return Graph(parents, GraphSampler.get_distribution(variables, parents))

	@staticmethod
	def to_file(graph):
		# TODO : Gaurav
		pass

	@staticmethod
	def from_file(filepath):
		# TODO : Gaurav
		pass