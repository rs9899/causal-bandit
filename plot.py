"""
Script for generating all plots.
"""

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import os

from graph_utilities import set_random_seed
from graph_generator import GraphSampler
import agents

# set plot theme
sn.set_style("darkgrid")

# parameters
N_RANDOM_SEEDS = 20
HORIZON        = 10000
STEP_SIZE      = 50

AGENTS = {
	'UCB'             : agents.UCBAgent,
	'KL-UCB'          : agents.KL_UCBAgent,
	'TS'              : agents.TSAgent,
	'OC-TS'           : agents.OC_TSAgent,
	'$\\epsilon$-greedy': agents.E_graphAgent
}

GRAPHS = [
	'graphs/graph-1.txt',
	# 'graphs/graph-2.txt',
	# 'graphs/graph-3.txt',
	# 'graphs/graph-4.txt'
]

for graph_file in GRAPHS:
	# G = GraphSampler.from_file(graph_file)
	G = GraphSampler.random_graph(7)
	A = sum([[{i:0}, {i:1}] for i in range(len(G.variables)-1)], [])
	mu_star = G.mu_star(A)
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

	# plot graph of estimated mu_star vs horizon
	plt.figure()
	plt.title(graph_file)
	plt.xlabel('Horizon')
	plt.ylabel('$\\sum_{t=1}^T r_t\\ /\\ T$')
	idx2name = [x for x in AGENTS.keys()]
	steps = np.arange(STEP_SIZE-1, HORIZON, STEP_SIZE)
	plt.xlim([steps[0], steps[-1]])
	for i in range(means.shape[0]):
		plt.plot(steps, means[i, :]/steps, label=idx2name[i])
		plt.fill_between(steps, (means[i,:]-stddevs[i,:])/steps, (means[i,:]+stddevs[i,:])/steps, alpha=0.2)
	plt.axhline(y=mu_star, color='r', linestyle='--', label='$\\mu*$')
	plt.legend()
	plt_name = os.path.splitext(os.path.basename(graph_file))[0]
	plt.savefig('mu_%s.pdf' % plt_name, bbox_inches='tight', dpi=300)

	# plot graph of regret vs horizon
	plt.figure()
	plt.title(graph_file)
	plt.xlabel('Horizon')
	plt.ylabel('Regret')
	idx2name = [x for x in AGENTS.keys()]
	steps = np.arange(STEP_SIZE-1, HORIZON, STEP_SIZE)
	plt.xlim([steps[0], steps[-1]])
	for i in range(means.shape[0]):
		plt.plot(steps, mu_star*steps-means[i, :], label=idx2name[i])
		plt.fill_between(steps, mu_star*steps-means[i, :]-stddevs[i,:], mu_star*steps-means[i, :]+stddevs[i,:], alpha=0.2)
	plt.axhline(y=0, color='r', linestyle='--')
	plt.legend()
	plt_name = os.path.splitext(os.path.basename(graph_file))[0]
	plt.savefig('regret_%s.pdf' % plt_name, bbox_inches='tight', dpi=300)