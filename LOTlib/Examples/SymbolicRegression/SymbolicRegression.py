# -*- coding: utf-8 -*-
"""
	A simple symbolic regression demo
"""

from Shared import *
from math import sin

CHAINS = 4
STEPS = 50000
SKIP = 0
data_sd = 0.10 # the SD of the likelihood

## The target function for symbolic regression 
target = lambda x: x + sin(1.0/x)

# Make up some learning data for the symbolic regression
def generate_data(data_size):
	
	# initialize the data
	data = []
	for i in range(data_size): 
		x = random()
		data.append( FunctionData(input=[x], output=target(x), ll_sd=data_sd) )
	
	return data
	
# generate some data
data = generate_data(50) # how many data points?

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Multicore, parallel

#one run with these parameters
def run(*args):
	# starting hypothesis -- here this generates at random
	h0 = GaussianLOTHypothesis(grammar)
	
	# We store the top 100 from each run
	fs = FiniteBestSet(10, max=True, key="posterior_score") 
	fs.add(  mh_sample(h0, data, STEPS, skip=SKIP)  )
	
	return fs
	
finitesample = FiniteBestSet(max=True) # the finite sample of all
results = map(run, [ [] ] * CHAINS ) # a not parallel
finitesample.merge(results)

for r in finitesample.get_all():
	print r.posterior_score, r.prior, r.likelihood, qq(str(r))
