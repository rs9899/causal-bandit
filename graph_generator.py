"""
Class for generating random graphs of varying topology.
"""

from copy import deepcopy
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
	def to_file(graph, filename):
		file1 = open(filename,"w")
		file1.write(str(len(graph.variables))+'\n')
		for v in graph.variables:
			file1.write(str(v)+'\n')
			file1.writelines([str(x) + " " for x in graph.parents[v]])
			file1.write('\n')
			if len(graph.parents[v])==0:
				file1.write(str(graph.distribution[v])+'\n')
				continue
			l = deepcopy(graph.distribution[v])
			l = l.reshape(-1)
			file1.writelines([str(x) + " " for x in l])
			file1.write('\n')
		file1.close()

	@staticmethod
	def from_file(filepath):
		file1 = open(filepath,"r")
		num_variables = int(file1.readline().split()[0])
		parents = [[] for _ in range(num_variables)]
		distribution = {}
		for i in range(0,num_variables):
			v = int(file1.readline().split()[0])
			parents[v] = np.asarray([int(x) for x in file1.readline().split()])
			if len(parents[v])==0:
				distribution[v] = float(file1.readline().split()[0])
				continue
			distribution[v] = np.asarray([float(x) for x in file1.readline().split()])
			distribution[v] = distribution[v].reshape([2]*len(parents[v]))
		file1.close()
		return Graph(parents, distribution)

