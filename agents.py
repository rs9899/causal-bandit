"""
Class descriptions for all agents.
"""

import numpy as np
import random
from tqdm import tqdm

class Agent(object):
	def __init__(self, G, A):
		self.graph = G
		self.actions = A
		self.numVar = len(self.graph.variables) - 1
		self.rewardVariable = self.numVar

	def _step(self, t=0):
		raise NotImplementedError

	def run(self, horizon=100, step_size=5):
		self.rewards = np.zeros(len(self.actions))
		self.n_pulled = np.zeros(len(self.actions))
		ans = []
		for t in tqdm(range(horizon)):
			self._step(t)
			if t % step_size == step_size - 1:
				ans.append(self.rewards.sum())
		return ans

class UCBAgent(Agent):
	def __init__(self, G, A):
		super(UCBAgent, self).__init__(G, A)

	def _step(self, t):
		if t < len(self.actions):
			# sample each arm/action once
			assignments = self.graph.intervention(self.actions[t])
			reward = assignments[len(assignments)-1]
			self.n_pulled[t] += 1
			self.rewards[t] += reward
		else:
			# follow UCB algorithm
			ucb = (self.rewards * 1.0 / self.n_pulled ) + np.sqrt( 2 * np.log(t) / self.n_pulled )
			a = np.argmax(ucb)
			assignments = self.graph.intervention(self.actions[a])
			reward = assignments[len(assignments)-1]
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
			reward = assignments[len(assignments)-1]
			self.n_pulled[t] += 1
			self.rewards[t] += reward
		elif len(self.actions) <= 2 and t < 2*len(self.actions):
			# initialize by pulling all arms twice
			# done to avoid ln(t)+3*ln(ln(t)) from becoming negative
			assignments = self.graph.intervention(self.actions[t%len(self.actions)])
			reward = assignments[len(assignments)-1]
			self.n_pulled[t%len(self.actions)] += 1
			self.rewards[t%len(self.actions)] += reward
		else:
			ucb = self._objective(t)
			arm = np.argmax(ucb)
			assignments = self.graph.intervention(self.actions[arm])
			reward = assignments[len(assignments)-1]
			self.n_pulled[arm] += 1
			self.rewards[arm] += reward

class TSAgent(Agent):
	def __init__(self, G, A):
		super(TSAgent, self).__init__(G, A)

	def _step(self, t):
		x = np.random.beta(1+self.s, 1+self.f)
		arm = np.argmax(x)
		assignments = self.graph.intervention(self.actions[arm])
		reward = assignments[len(assignments)-1]
		self.n_pulled[arm] += 1
		self.s[arm] += reward; self.f[arm] += 1-reward

	def run(self, horizon=100, step_size=5):
		self.s = np.zeros(len(self.actions), dtype=int)
		self.f = np.zeros(len(self.actions), dtype=int)
		self.n_pulled = np.zeros(len(self.actions), dtype=int)
		ans = []
		for t in tqdm(range(horizon)):
			self._step(t)
			if t % step_size == step_size - 1:
				ans.append(self.s.sum())
		return ans

class OC_TSAgent(Agent):
	def __init__(self, G, A):
		super(OC_TSAgent, self).__init__(G, A)

	def _step(self, t):
		success_chance = np.zeros(len(self.actions))
		for a in range(len(self.actions)):
			partition_prob = np.random.dirichlet(self.dirc[:,a]).reshape(-1,1)
			sample_prob = np.random.beta(self.beta[:,0], self.beta[:,1]).reshape(1,-1)
			success_chance[a] = (sample_prob @ partition_prob).item()
		
		arm = np.argmax(success_chance)
		assignments = self.graph.intervention(self.actions[arm])
		reward = assignments[len(assignments)-1]

		z = sum([2**i * assignments[i] for i in range(len(self.graph.variables)-1)])

		self.dirc[z, arm] += 1
		self.beta[z, 1-int(reward)] += 1
		self.rewards[arm] += reward

	def run(self, horizon=100, step_size=5):
		n_part = 2 ** (len(self.graph.variables) - 1)
		self.beta = np.ones([n_part, 2], dtype=int)
		self.dirc = np.ones([n_part, len(self.actions)], dtype=int)
		self.rewards = np.zeros(len(self.actions))
		ans = []
		for t in tqdm(range(horizon)):
			self._step(t)
			if t % step_size == step_size - 1:
				ans.append(self.rewards.sum())
		return ans

class EpsilonAgent(Agent):
	def __init__(self, G, A):
		super(EpsilonAgent, self).__init__(G, A)

		# This stores the intervention and rewards in a dictionary format:
		# self.run_history["000010"] = 3 indicates that values
		# [x1, x2, x3, x4, x5] = [0, 0, 0, 0, 1] led to a reward of 0 (the 
		# last character in the string) in exactly 3 runs
		self.run_history = dict()

		# Total number of actions which led to a reward of 1 and total 
		# number of actions
		self.total_successful_actions = 0
		self.total_actions = 0


	# Return a list of assignments to variables in graph given an encoding
	# of the assignments in a string format
	def _getAssignmentFromString(self, sx):
		return list(map(int, sx))

	# Return an encoding of the assignments in a string format given a 
	# list of assignments to variables in graph
	def _getStringFromAssignment(self, assignment):
		# If the input is a dictionary (i.e. an assignment dictionary)
		#
		# Can be sped up using the invariant mapping present in 
		# assignment.values()
		if type(assignment) is dict:
			string = [0] * len(self.graph.variables)
			for key in assignment.keys():
				string[int(key)] = assignment[key]

			return "".join(map(str, string))

		# If input is a list of assignments
		if type(assignment) is list:
			return "".join(map(str, assignment))

		raise NotImplementedError

	# Update self.prob_successful_action and self.prob_action given the
	# sample 'assignment'
	def _updateProbabilities(self, assignment):
		actions = self.actions
		reward = assignment[self.rewardVariable]

		for key in assignment.keys():
			self.total_actions += 1

			if reward == 1:
				self.total_successful_actions += 1

		dict_index = self._getStringFromAssignment(assignment)

		if dict_index not in self.run_history.keys():
			self.run_history[dict_index] = 1
		else:
			self.run_history[dict_index] += 1

	# Run a single iteration
	def _step(self, time_step, epsilon):
		# Explore different actions
		actions = self.actions
		if random.random() < epsilon:
			i = int(random.random() * len(self.actions))
			assignment = self.graph.intervention(actions[i])
			
			self._updateProbabilities(assignment)

		# Exploit using run history table
		else:
			
			expectations = [0] * len(actions)

			for x in actions:
				var = list(x.keys())[0]
				action = x[var]

				assignment_count = dict()
				reward_count = dict()
				consistent_assn_count = dict()
				run_history = self.run_history
				for key in run_history.keys():
					parent_assignment = [
						i for i in key 
						if i in self.graph.parents[var]
					]
					parent_assignment = "".join(map(str, parent_assignment))
					
					if parent_assignment in assignment_count.keys():
						assignment_count[parent_assignment] += 1
					else:
						assignment_count[parent_assignment] = 1
						reward_count[parent_assignment] = 0
						consistent_assn_count[parent_assignment] = 0
				
				for key in assignment_count.keys():
					if consistent_assn_count[key] != 0:
						expectations[2*var+action] += (reward_count[key] / consistent_assn_count[key]) * (assignment_count[key] * self.total_actions)

			action = expectations.index(max(expectations))
			assignment = self.graph.intervention({var: action})
			self._updateProbabilities(assignment)

	# Run the algorithm for given horizons
	def run(self, horizon=100, step_size = 5):
		def positiveReward(sx):
			# assignments = _getAssignmentFromString(sx)
			# return assignments[-1] == 1
			return sx[-1] == "1"
			
		ans = []
		for t in tqdm(range(horizon)):
			self._step(t, epsilon=0.2)
			if t % step_size == step_size - 1:
				reward_counts = [
					self.run_history[assn]
					for assn in self.run_history.keys()
					if positiveReward(assn)
					]
				ans.append(sum(reward_counts))
		return ans

class SampleGraph:
	def __init__(self,G):
		random.seed(4)
		self.variables = np.arange(len(G.parents))
		self.parents = G.parents
		self.ZeroCount = {}
		self.OneCount = {}
		for i in self.variables:
			self.ZeroCount[i] = np.zeros([2**len(self.parents[i]),]) + 1
			self.OneCount[i] = np.zeros([2**len(self.parents[i]),]) + 1

	def update(self,assignment,varIntervened = []):
		for i in self.variables:
			if i not in varIntervened:
				idx = 0
				j = 0
				for p in self.parents[i]:
					idx = idx + (assignment[p] * (2**j) )
					j = j + 1
				if assignment[i] == 0:
					self.ZeroCount[i][idx] += 1
				else:
					self.OneCount[i][idx] += 1

	def binaryIntervention(self, assignment = {}):
		returnAssign = {}
		for v in self.variables:
			if v not in assignment:
				idx = 0
				j = 0
				for p in self.parents[v]:
					idx = idx + (returnAssign[p] * (2**j) )
					j = j + 1
				p = self.ZeroCount[v][idx] * 1.0 / (self.OneCount[v][idx] + self.ZeroCount[v][idx])
				if random.random() < p:
					returnAssign[v] = 0
				else:
					returnAssign[v] = 1
			else:
				returnAssign[v] = assignment[v]

		return returnAssign[ len(self.variables) - 1 ]
				


class E_graphAgent(Agent):
	def __init__(self,G,A,epsilon = 0.1,step = 10):
		super(E_graphAgent, self).__init__(G, A)
		self.epsilon = epsilon
		self.step = step

	def _step(self,time_step,epsilon):
		# WRITE CODE FOR EPSILON DECAY >>> LESS EXPLORATION

		actions = self.actions
		if random.random() < epsilon:
			bestAction = int(random.random() * len(actions))
			assignment = self.graph.intervention(actions[bestAction])
		else:
			bestAction = 0
			rewardArray = []
			for action in actions:
				reward = 0
				for i in range(int(1e2)):
					reward += self.myGraph.binaryIntervention(action)
				rewardArray.append(reward)
			bestAction = np.argmax(np.asarray(rewardArray))
			assignment = self.graph.intervention(actions[bestAction])
		
		self.rewards[bestAction] += assignment[self.rewardVariable]

		self.myGraph.update(assignment , list(actions[bestAction].keys()) )


	def run(self,horizon=100,step_size=5):
		self.myGraph = SampleGraph(self.graph)
		self.numAction = len(self.actions)
		self.rewards = np.zeros(self.numAction)
		ans = []
		for t in tqdm(range(horizon)):
			self._step(t,self.epsilon)
			if t%step_size==step_size-1:
				cum_award = self.rewards.sum()
				ans.append(cum_award)
				# print(cum_award)
		return ans
