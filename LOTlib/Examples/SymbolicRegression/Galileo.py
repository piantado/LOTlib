# -*- coding: utf-8 -*-
from Shared import *

"""
	This uses Galileo's data on a falling ball. See: http://www.amstat.org/publications/jse/v3n1/datasets.dickey.html
	See also, Jeffreys, W. H., and Berger, J. O. (1992), "Ockham's Razor and Bayesian Analysis," American Scientist, 80, 64-72 (Erratum, p. 116). 
"""

# NOTE: these must be floats, else we get hung up on powers of ints
data = [ 
         FunctionData(args=[1000.], output=1500.),\
         FunctionData(args=[828.], output=1340.),\
         FunctionData(args=[800.], output=1328.),\
         FunctionData(args=[600.], output=1172.),\
         FunctionData(args=[300.], output=800.), \
         FunctionData(args=[0.], output=0.) # added 0,0 since it makes physical sense. 
	]
	
CHAINS = 10
STEPS = 1000000
SKIP = 0
LL_SD = 50.0
PRIOR_TEMPERATURE=1.0

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the running function

def run(*args):
	print "Running "
	
	# We store the top 100 from each run
	pq = FiniteBestSet(100, max=True) 
	
	# starting hypothesis -- here this generates at random
	initial_hyp = GaussianStandardExpression(G, prior_temperature=PRIOR_TEMPERATURE, ll_sd=LL_SD)

	# populate the finite sample by running the sampler for this many steps
	for x in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, STEPS, skip=SKIP):
		pq.push(x, x.lp)
		#print x.lp, x.prior, x.likelihood, q(x)
	
	return pq

finitesample = FiniteBestSet(max=True) # the finite sample of all
results = map(run, [ [None] ] * CHAINS ) # Run on a single core
finitesample.merge(results)
	
## and display
for r in finitesample.get_sorted(decreasing=True):
	print r.lp, r.prior, r.likelihood, "\t", q(str(r))
	
	