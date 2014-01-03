# -*- coding: utf-8 -*-
"""
	A simple symbolic regression demo
"""

from Shared import *
from math import sin

CHAINS = 4
STEPS = 50000
SKIP = 0
LL_SD = 0.10 # the SD of the likelihood

## The target function for symbolic regression 
target = lambda x: x + sin(1.0/x)

# Make up some learning data for the symbolic regression
def generate_data(data_size):
	
	# initialize the data
	data = []
	for i in range(data_size): 
		x = random()
		data.append( FunctionData(args=[x], output=target(x), ll_sd=LL_SD) )
	
	return data
	
# generate some data
data = generate_data(50) # how many data points?
#print "# DATA = ", data

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Multicore, parallel

#one run with these parameters
def run(*args):
	print "Running new chain."
	
	# We store the top 100 from each run
	fs = FiniteBestSet(10, max=True) 
	
	# starting hypothesis -- here this generates at random
	initial_hyp = GaussianLOTHypothesis(G)

	# populate the finite sample by running the sampler for this many steps
	for x in mh_sample(initial_hyp, data, STEPS, skip=SKIP):
		fs.push(x, x.lp)
		print x.lp, x.prior, x.likelihood, q(x)
	
	return fs
	
finitesample = FiniteBestSet(max=True) # the finite sample of all
#results = parallel_map(run, [ [] ] * CHAINS ) # a parallel computed array of finite samples
results = map(run, [ [] ] * CHAINS ) # a not parallel
finitesample.merge(results)

for r in finitesample.get_all():
	print r.lp, r.prior, r.likelihood, "\t", q(str(r))