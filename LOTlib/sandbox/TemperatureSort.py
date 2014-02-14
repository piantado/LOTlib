	
"""
	Experimental Try out a parallel tempering scheme where we sort the chains every now and then

"""
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import *
import numpy
import math
from random import shuffle
import LOTlib.Miscellaneous
from LOTlib.Examples.Number.Shared import *



temperatures = numpy.arange(1.0,2.0,0.1)
N = len(temperatures)
rN = range(N)
hypotheses = [   NumberExpression(G) for i in xrange(N) ]

lNfac = sum([ log(i+1) for i in xrange(N)])

STEPS = 100
DATA_SIZE = 1000
OUTER_IT = 100

data = generate_data(DATA_SIZE)



for outerit in xrange(OUTER_IT):
	if LOTlib.SIG_INTERRUPTED: break
	
	# Update each chain
	for i in xrange(N):
		for s in LOTlib.MetropolisHastings.mh_sample(hypotheses[i], data, STEPS, temperature=temperatures[i]):
			hypotheses[i] = s

	# Now a temperature chain proposal:
	fb = 0.0
	if flip(0.5): # randomly shuffle
		print "Shuffle proposal"
		shuffle(rN)
		proposal = [ hypotheses[ii] for ii in rN ]
		
		is_sorted  = all([ proposal[i].lp < proposal[i+1].lp for i in xrange(N-1) ])
	else: # sort via the posterior
		print "Sort proposal"
		
		proposal = sorted(hypotheses, key=lambda x: x.lp, reverse=True)
		
		is_sorted = True
		
	was_sorted = all([ hypotheses[i].lp < hypotheses[i+1].lp for i in xrange(N-1) ])

	# If we are sorted now, it happened with probability 1/2*1.0 + (1/2)*1/(N!)
	# if not, it had probability (1/2)*(1/N!)
	if is_sorted: f = log(0.5 + 0.5 * exp(-lNfac)) # TODO: use log1+x primitives here
	else:         f = log(0.5) - lNfac

	if was_sorted: b = log(0.5 + 0.5 * exp(-lNfac)) # TODO: use log1+x primitives here
	else:          b = log(0.5) - lNfac

	fb = f-b

	proposal_lp  = sum([ x.lp / t for x,t in zip( proposal, temperatures) ])
	hypotheses_lp = sum([ x.lp / t for x,t in zip( hypotheses, temperatures) ])

	r =proposal_lp - hypotheses_lp - fb 
	if r > 0 or random() < exp(r):
		hypotheses = proposal
		print "ACCEPT"
		
	print proposal_lp, hypotheses_lp, fb














	