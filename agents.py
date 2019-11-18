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

		# This stores the intervention and rewards in a dictionary format:
		# self.run_history["000010"] = 3 indicates that values
		# [x1, x2, x3, x4, x5] = [0, 0, 0, 0, 1] led to a reward of 0 (the 
		# last character in the string) in exactly 3 runs
		self.run_history = dict()

	# Return a list of assignments to variables in graph given an encoding
	# of the assignments in a string format
	def _getAssignmentFromString(self, sx):
		return list(map(int, sx))

	# Return an encoding of the assignments in a string format given a 
	# list of assignments to variables in graph
	def _getStringFromAssignment(self, sx):
		return "".join(map(str, sx))

	def _step(self, time_step, epsilon):
		# Explore different actions
		if random.random() < epsilon:
			i = int(random.random * len(self.actions))
			assignment = self.graph.intervention(actions[i])
			

		# Exploit actions
		raise NotImplementedError

	def run(self, horizon=100):

		def positiveReward(sx):
			# assignments = _getAssignmentFromString(sx)
			# return assignments[-1] == 1
			return sx[-1] == "1"

		for t in range(horizon):
			self._step(t, epsilon=0.2)

		# Compile all runs which resulted in a positive record
		reward_counts = [
			self.run_history[assn]
			for assn in self.run_history.keys()
			if positiveReward(assn)
			]
		
		return sum(reward_counts)