import numpy as np 
import random
random.seed(40)


from matplotlib import pyplot as plt

from graph_utilities import graph 
from graph_generator import graph_samples
from agents import *



n = 10
g = graph_samples(n).linear_graph()
a = []
for i in range(n-1):
	a.append({i:0})
	a.append({i:1})

# ucb_agent = UCBAgent(g, a)
# ucb_rewards = ucb_agent.run(100000,1000)
# x = []
# for i in range(100):
# 	x.append(1000*(i+1))

# print(ucb_rewards)
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])
# # plt.show()


# ucb_agent = KL_UCBAgent(g, a)
# ucb_rewards = ucb_agent.run(100000,1000)
# x = []
# for i in range(100):
# 	x.append(1000*(i+1))

# print(ucb_rewards)
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])


# ucb_agent = TSAgent(g, a)
# ucb_rewards = ucb_agent.run(100000,1000)
# x = []
# for i in range(100):
# 	x.append(1000*(i+1))

# print(ucb_rewards)
# plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])


ucb_agent = EpsilonAgent(g, a)
ucb_rewards = ucb_agent.run(1000)
x = []
for i in range(100):
	x.append(10*(i+1))

print(ucb_rewards)
plt.plot(x, [ucb_rewards[i]/x[i] for i in range(len(x))])







plt.show()


