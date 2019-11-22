"""
Bayesian graph with topologically sorted variables.
"""

import numpy as np 
import random

def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)

class Graph(object):
	def __init__(self, parents, distribution):
		self.variables = np.arange(len(parents))
		self.numVar = len(self.variables) - 1
		self.rewardVariable = self.numVar
		self.parents = parents
		self.distribution = distribution

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

	def allPossibleAssign(self,lis):
		if len(lis) == 0:
			return [{}]
		Z = self.allPossibleAssign(lis[1:])
		l = []
		for z in Z:
			v = {lis[0] : 0}
			u = {lis[0] : 1}
			v.update(z)
			u.update(z)
			l.append(v)
			l.append(u)
		return l


	def P_helper(self,X,vals,A):
		if len(X) == 0:
			return 1.0
		var = X[0]
		if var in A:
			if vals[var] == A[var]:
				return self.P_helper(X[1:], vals, A)
			else:
				return 0.0
		pa_var = self.parents[var]
		if len(pa_var) == 0:
			p = self.distribution[var][0]
			if vals[var] == 0:
				p = 1 - p
			return p * self.P_helper(X[1:], vals, A)
		
		new_var = set(pa_var).union(set(X[1:]))
		pa_assign = self.allPossibleAssign(pa_var)
		valid_assign = [z for z in pa_assign if all([z[i] == v for i,v in vals.items() if i in z])]

		prob = 0.0
		for z in valid_assign:
			prob_given_parent  = self.distribution[var] 
			for par in self.parents[var]:
				prob_given_parent = prob_given_parent[z[par]]
			if vals[var] == 0:
				prob_given_parent = 1 - prob_given_parent

			new_vals = z
			new_vals.update(vals)

			prob += (prob_given_parent * self.P_helper(list(new_var), new_vals, A) )

		return prob



	def mu_star(self,actions):
		return max([
			self.P_helper(
					[self.rewardVariable],
					{self.rewardVariable : 1},
					action
					) for action in actions	])		
