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
		raise NotImplementedError

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

	def run(self, horizon=100):
		self.rewards = np.zeros(len(A))
		self.n_pulled = np.zeros(len(A))
		for t in range(horizon):
			self._step(t)
		return self.rewards.sum()

class EpsilonAgent(Agent):
	def __init__(self, G, A):
		super(EpsilonAgent, self).__init__(G, A)

	def _step(self, t):
		raise NotImplementedError

	def run(self, horizon=100):
		raise NotImplementedError