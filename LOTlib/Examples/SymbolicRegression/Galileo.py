# -*- coding: utf-8 -*-
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.DataAndObjects import FunctionData
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import qq
from Grammar import grammar

"""
	This uses Galileo's data on a falling ball. See: http://www.amstat.org/publications/jse/v3n1/datasets.dickey.html
	See also, Jeffreys, W. H., and Berger, J. O. (1992), "Ockham's Razor and Bayesian Analysis," American Scientist, 80, 64-72 (Erratum, p. 116). 
"""

# NOTE: these must be floats, else we get hung up on powers of ints
data_sd = 50.0
data = [FunctionData(input=[1000.], output=1500., ll_sd=data_sd),
        FunctionData(input=[828.], output=1340., ll_sd=data_sd),
        FunctionData(input=[800.], output=1328., ll_sd=data_sd),
        FunctionData(input=[600.], output=1172., ll_sd=data_sd),
        FunctionData(input=[300.], output=800., ll_sd=data_sd), 
        FunctionData(input=[0.], output=0., ll_sd=data_sd) # added 0,0 since it makes physical sense. 
]
	
CHAINS = 10
STEPS = 10000000
SKIP = 0
PRIOR_TEMPERATURE=1.0

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the running function

def run(*args):
	
	# starting hypothesis -- here this generates at random
	h0 = GaussianLOTHypothesis(grammar, prior_temperature=PRIOR_TEMPERATURE)
	
	# We store the top 100 from each run
	pq = FiniteBestSet(100, max=True, key="posterior_score") 
	pq.add( mh_sample(h0, data, STEPS, skip=SKIP)  )
		
	return pq

finitesample = FiniteBestSet(max=True) # the finite sample of all
results = map(run, [ [None] ] * CHAINS ) # Run on a single core
finitesample.merge(results)
	
## and display
for r in finitesample.get_all(decreasing=False, sorted=True):
	print r.posterior_score, r.prior, r.likelihood, qq(str(r))
