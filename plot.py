"""
Script for generating all plots.
"""

import matplotlib.plt as pyplot
import seaborn as sn
import numpy as np

from graph_utilities import set_random_seed
from graph_generator import GraphSampler
import agents

N_RANDOM_SEEDS = 10
HORIZON        = 10000
STEPS          = 500

AGENTS = {
	'UCB'             : agents.UCBAgent,
	'KL-UCB'          : agents.KL_UCBAgent,
	'TS'              : agents.TSAgent,
	'OC-TS'           : agents.OC_TSAgent,
	'\\epsilon-greedy': agents.E_graphAgent
}

GRAPHS = [
	'graphs/graph-1.txt',
	'graphs/graph-2.txt',
	'graphs/graph-3.txt',
	'graphs/graph-4.txt'
]

for graph_file in GRAPHS:
	G = GraphSampler.from_file(graph_file)
	A = sum([[{i:0}, {i:1}] for i in range(len(G.variables))-1], [])
	results = np.empty([len(AGENTS), N_RANDOM_SEEDS, int(np.ceil(HORIZON / STEPS))])
	
	for name in AGENTS:
		