# -*- coding: utf-8 -*-
"""
	A quick demo of the number model. 
	
	Note: CTRL-C breaks out of the MCMC loop, and the processes at the bottom with average likelihood for each hypothesis. 
"""

from Shared import *

LARGE_DATA_SIZE = 10000 # this is what we compute the average LL on
DATA_SIZE = 300
TRACE = True
STEPS = 1000000 
SKIP = 1

# # # # # # # # # # # # # # # # # # # # # # # # #
# Generate some data

data = generate_data(DATA_SIZE)

# A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
initial_hyp = NumberExpression(G)

# store hypotheses we've found
allhyp = FiniteBestSet(max=True,N=1000)

#from LOTlib.Memoization.Memoizer import BoundedMemoize

import LOTlib.Inference
import LOTlib.Inference.TemperedTransitions

# A bunch of different MCMC algorithms to try. mh_sample is from the Rational Rules paper and generally works very well. 
#for h in  LOTlib.Inference.TemperedTransitions.tempered_transitions_sample(initial_hyp, data, 500000, skip=0, temperatures=[1.0, 1.25, 1.5]):		
#for h in  LOTlib.Inference.ParallelTempering.parallel_tempering_sample(initial_hyp, data, STEPS, within_steps=10, yield_all=True, temperatures=[1.0,1.05, 1.1]):
for h in LOTlib.Inference.MetropolisHastings.mh_sample(initial_hyp, data, STEPS, skip=SKIP):
	if TRACE: 
		print q(get_knower_pattern(h)), h.compute_prior(), h.compute_likelihood(data), q(h)
		
	# add h to our priority queue, with priority of its log probability, h.posterior_score
	allhyp.push(h, h.posterior_score)

# save these hypotheses

from LOTlib.Serialization import serialize2file
serialize2file(allfs, "demo-hypotheses.pkl") # save this in a file

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## now re-evaluate everything we found on new data
#huge_data = generate_data(LARGE_DATA_SIZE)
	
#save this with a huge data set -- eval with average ll 
#H = allhyp.get_sorted()

# compute the posterior for each hypothesis
#[ h.compute_posterior(huge_data) for h in H]
	
# show the *average* ll for each hypothesis, at this data size
#for h in H:
	#print h.prior, h.likelihood/float(LARGE_DATA_SIZE), q(get_knower_pattern(h)),  q(h) # a quoted x




