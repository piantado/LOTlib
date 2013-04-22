# -*- coding: utf-8 -*-

"""
	An experimental adaptive method, that runs a bunch of standard mcmc in order to find tree parts that tend to be used in 
	good grammars.
	
	These parts can be added to the PCFG for preferential generations in the PCFG

"""

from Number_Shared import *
import LOTlib.EnumerativeSearch

# # # # # # # # # # # # # # # # # # # # # # # # #
# Generate some data

data = generate_data(30)
initial_hyp = NumberExpression()

subtree_mean = dict()
subtree_count = dict()
seen = set()
for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, 50000, skip=0):
	
	if h.lp > float("-inf") and h not in seen:
		seen.add(h)
		
		for tt in h.value:
			#print tt
			subtree_mean[tt] = subtree_mean.get(tt, 0.0) + h.lp
			subtree_count[tt] = subtree_count.get(tt, 0) + 1

# compute the actual means:
for t in subtree_mean.keys():
	if subtree_count[t] > 10:
		print subtree_mean[t] / subtree_count[t], subtree_count[t],  t
