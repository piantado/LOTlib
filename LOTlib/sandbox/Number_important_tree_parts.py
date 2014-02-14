# -*- coding: utf-8 -*-

"""
	An experimental adaptive method, that runs a bunch of standard mcmc in order to find tree parts that tend to be used in 
	good grammars.
	
	These parts can be added to the PCFG for preferential generations in the PCFG

"""

from LOTlib.Examples.Number.Shared import *
from LOTlib.Inference.MetropolisHastings import mh_sample
from collections import defaultdict

# # # # # # # # # # # # # # # # # # # # # # # # #
# Generate some data

data = generate_data(30)
initial_hyp = NumberExpression(G)

# give me some hypotheses
hyps = set(mh_sample(initial_hyp, data, 10000, skip=0))

# and their normalizer
hypZ = logsumexp([ h.posterior_score for h in hyps])

# In the posterior distribution, how likely is any tree 
# expansion from a given nonterminal? Here we just add up posterior probabilities
# of all expansions
d = defaultdict(lambda : dict() )
for h in hyps:
	for tt in h.value:
		lp = h.posterior_score - hypZ
		d[tt.returntype][tt] = logsumexp([ d[tt.returntype].get(tt,float("-inf")), lp])

## and re-normalize
for k in d:
	z = logsumexp(d[k].values())
	for k2 in d[k].keys():
		d[k][k2] -= z

# And print out based on the difference
for nt in d:
	for t in d[nt]:
		
		priorp = t.log_probability()
		postp  = d[nt][t]
		
		print exp(postp) * (postp-priorp), priorp, postp, qq(t)

"""
	We really want the set of expansions that will move the prior most in the direction of the posterior

"""

