"""
Script for generating all plots.
"""

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from graph_utilities import set_random_seed
from graph_generator import GraphSampler
import agents

# set plot theme
sn.set_style("darkgrid")

# parameters
N_RANDOM_SEEDS = 3
HORIZON        = 10000
STEP_SIZE      = 500

AGENTS = {
	'UCB'             : agents.UCBAgent,
	'KL-UCB'          : agents.KL_UCBAgent,
	'TS'              : agents.TSAgent,
	'OC-TS'           : agents.OC_TSAgent,
	# '\\epsilon-greedy': agents.E_graphAgent
}

GRAPHS = [
	'graphs/graph-1.txt',
	# 'graphs/graph-2.txt',
	# 'graphs/graph-3.txt',
	# 'graphs/graph-4.txt'
]

for graph_file in GRAPHS:
	# G = GraphSampler.from_file(graph_file)
	G = GraphSampler.random_graph(5)
	A = sum([[{i:0}, {i:1}] for i in range(len(G.variables)-1)], [])
	results = np.empty([len(AGENTS), N_RANDOM_SEEDS, HORIZON // STEP_SIZE])

	for i, name in enumerate(AGENTS):
		agnt = AGENTS[name](G, A)
		for seed in range(N_RANDOM_SEEDS):
			set_random_seed(seed)
			temp_res = agnt.run(HORIZON, STEP_SIZE)
			results[i, seed, :] = np.array(temp_res)

	# compute mean and stddev
	means = np.mean(results, axis=1)
	stddevs = np.std(results, axis=1)

	# plot graph
	plt.figure()
	plt.title(graph_file)
	plt.xlabel('Horizon')
	plt.ylabel('E[Y=1 | do(X=a)]')
	idx2name = [x for x in AGENTS.keys()]
	steps = np.arange(STEP_SIZE-1, HORIZON, STEP_SIZE)
	for i in range(means.shape[0]):
		plt.plot(steps, means[i, :]/steps, label=idx2name[i])
		plt.fill_between(steps, (means[i,:]-stddevs[i,:])/steps, (means[i,:]+stddevs[i,:])/steps, alpha=0.2)
	plt.legend()
	plt.show()