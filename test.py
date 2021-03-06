import numpy as np 
import random
random.seed(40)


from matplotlib import pyplot as plt

from graph_utilities import Graph 
from graph_generator import GraphSampler
from agents import *

n = 5
g = GraphSampler.linear_graph(n)
a = []
for i in range(n-1):
	a.append({i:0})
	a.append({i:1})

m = g.mu_star(a)
print(m)
print(g.distribution)

# ucb_agent = TSAgent(g, a)
# ucb_rewards = ucb_agent.run(30000,1000)
# x = []
# for i in range(30):
# 	x.append(1000*(i+1))

# print(ucb_rewards[-1])
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])

# ucb_agent = OC_TSAgent(g, a)
# ucb_rewards = ucb_agent.run(30000,1000)
# x = []
# for i in range(30):
# 	x.append(1000*(i+1))

# print(ucb_rewards[-1])
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])

# ucb_agent = EpsilonAgent(g, a)
# ucb_rewards = ucb_agent.run(30000, 1000)

# x = []
# for i in range(30):
# 	x.append(1000*(i+1))

# print(ucb_rewards)
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])
# plt.show()

# ucb_agent = E_graphAgent(g, a  , switch = 0)
# ucb_rewards = ucb_agent.run(3000, 100)

# x = []
# for i in range(30):
# 	x.append(100*(i+1))

# print(ucb_rewards[-1])
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])

# ucb_agent = E_graphAgent(g, a , switch = 1)
# ucb_rewards = ucb_agent.run(3000, 100)

# x = []
# for i in range(30):
# 	x.append(100*(i+1))

# print(ucb_rewards[-1])
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])





# plt.show()




