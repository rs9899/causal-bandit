"""
Class descriptions for all agents.
"""

import numpy as np

class Agent(object):
	def __init__(self, G, A):
		self.graph = G
		self.actions = A

	def _step(self, t=0):
		raise NotImplementedError

	def run(self, horizon=100):
		self.rewards = np.zeros(len(A))
		self.n_pulled = np.zeros(len(A))
		for t in range(horizon):
			self._step(t)
		return self.rewards.sum()

class UCBAgent(Agent):
	def __init__(self, G, A):
		super(UCBAgent, self).__init__(G, A)

	def _step(self, t):
		if t < len(self.actions):
			# sample each arm/action once
			assignments = self.graph.intervention(self.actions[t])
			reward = assignments[len(assignments)]
			self.n_pulled[t] += 1
			self.rewards[t] += reward
		else:
			# follow UCB algorithm
			ucb = self.rewards/self.n_pulled + np.sqrt(2*np.log(t)/self.n_pulled)
			a = np.argmax(ucb)
			assignments = self.graph.intervention(self.actions[a])
			reward = assignments[len(assignments)]
			self.n_pulled[a] += 1
			self.rewards[a] += reward

class KL_UCBAgent(Agent):
	def __init__(self, G, A):
		super(KL_UCBAgent, self).__init__(G, A)

	def _objective(self, t):
		DELTA = 1e-4
		EPSILON = 1e-4

		p_hat = self.rewards / self.n_pulled
		t1 = np.array([0. if p_hat[i] == 0. or p_hat[i] == 1. \
			else p_hat[i] * np.log(p_hat[i]) + (1. - p_hat[i]) * np.log(1. - p_hat[i]) \
			for i in range(len(self.actions))])
		t2 = (np.log(t) + 3. * np.log(np.log(t))) / self.n_pulled
		constant = t1 - t2

		q = np.minimum(np.maximum((1. + p_hat) / 2., p_hat + DELTA), 1.-DELTA)
		fn_val = constant - p_hat*np.log(q) - (1. - p_hat)*np.log(1. - q)
		
		while not all(np.logical_or(np.abs(fn_val) < EPSILON, q >= 1.-DELTA)):
			fn_q = (q - p_hat) / (q * (1. - q))
			q -= fn_val / fn_q
			q = np.minimum(np.maximum(q, p_hat + DELTA), 1.-DELTA)
			fn_val = constant - p_hat*np.log(q) - (1. - p_hat)*np.log(1. - q)

		return q

	def _step(self, t):
		if len(self.actions) > 2 and t < len(self.actions):
			# initialize by pulling all arms once
			assignments = self.graph.intervention(self.actions[t])
			reward = assignments[len(assignments)]
			self.n_pulled[t] += 1
			self.rewards[t] += reward
		elif len(self.actions) <= 2 and t < 2*len(self.actions):
			# initialize by pulling all arms twice
			# done to avoid ln(t)+3*ln(ln(t)) from becoming negative
			assignments = self.graph.intervention(self.actions[t%len(self.actions)])
			reward = assignments[len(assignments)]
			self.n_pulled[t%len(self.actions)] += 1
			self.rewards[t%len(self.actions)] += reward
		else:
			ucb = self._objective(t)
			arm = np.argmax(ucb)
			assignments = self.graph.intervention(self.actions[arm])
			reward = assignments[len(assignments)]
			self.n_pulled[arm] += 1
			self.rewards[arm] += reward

class TSAgent(Agent):
	def __init__(self, G, A):
		super(TSAgent, self).__init__(G, A)

	def _step(self, t):
		x = np.random.beta(1+self.s, 1+self.f)
		arm = np.argmax(x)
		assignments = self.graph.intervention(self.actions[arm])
		reward = assignments[len(assignments)]
		self.n_pulled[arm] += 1
		self.s[arm] += reward; self.f[arm] += 1-reward

	def run(self, horizon=100):
		self.s = np.zeros(len(self.actions), dtype=int)
		self.f = np.zeros(len(self.actions), dtype=int)
		self.n_pulled = np.zeros(len(self.actions), dtype=int)
		for t in range(horizon):
			self._step(t)
		return self.s.sum()

class OC_TSAgent(Agent):
	def __init__(self, G, A):
		super(OC_TSAgent, self).__init__(G, A)

	def _step(self):
		succesChance = np.zeros( self.numAction )
		for a in range( self.numAction ):
			partionProb = np.random.dirichlet(self.dirch[:,a]).reshape((self.numPartition,1))
			sampl = np.random.beta(self.beta[:,0],self.beta[:,1]).reshape((1 , self.numPartition))
			succesChance[a] = np.asscalar(sampl @ partionProb)
		
		best = np.argmax(succesChance)
		d = self.actions[best]
		d = self.graph.intervention(d) 
		r = d[self.numAction]
		z = 0
		for i in range(self.numVar):
			z = z + ( 2**i * d[i])

		self.dirch[z,best] += 1
		self.rewards[best] += r
		if r == 1:
			self.beta[z,0] += 1
		else:
			self.beta[z,1] += 1

	def run(self, horizon=100):
		self.numAction = len(self.actions)
		self.numVar = len(self.graph.variables) - 1
		self.numPartition = 2 ** (self.numVar)
		self.beta = np.zeros((self.numPartition,2), dtype=int) + 1
		self.dirch = np.zeros((self.numPartition, self.numAction), dtype=int) + 1
		self.rewards = np.zeros(self.numAction)
		for t in range(horizon):
			self._step(t)
		return self.rewards.sum()



class SampleGraph:
	def __init__(self,G):
		random.seed(4)
		self.variables = np.arange(len(G.parents))
		self.parents = G.parents
		self.ZeroCount = {}
		self.OneCount = {}
		for i in self.variables:
			self.ZeroCount[i] = np.random.zeros(2**len(self.parents[i])) + 1
			self.OneCount[i] = np.random.zeros(2**len(self.parents[i])) + 1

	def update(self,assignment,varIntervened = []):
		for i in self.variables:
			if i not in varIntervened:
				idx = 0
				j = 0
				for p in self.parents[i]:
					idx = idx + (assignment[p] * (2**j) )
					j = j + 1
				if assignment[i] == 0:
					ZeroCount[idx] += 1
				else:
					OneCount[idx] += 1

	def sampleIntervention(self, assignment = []):
		for v in self.variables:
			if v not in assignment:
				idx = 0
				j = 0
				for p in self.parents[i]:
					idx = idx + (assignment[p] * (2**j) )
					j = j + 1
				p = ZeroCount[idx] * 1.0 / (OneCount[idx] + ZeroCount[idx])
				if random.random() < x[0]:
					assignment[v] = 0
				else:
					assignment[v] = 1
				



class E_graphAgent(Agent):
	def __init__(self,G,A):
		super(E_graphAgent, self).__init__(G, A)

	def _step(self):
		return

	def run(self,horizon=100):
		self.myGraph = SampleGraph(self.graph)