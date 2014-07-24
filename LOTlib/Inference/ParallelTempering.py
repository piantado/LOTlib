"""
	TODO: RECAST THIS IN THE OO FORMAT OF MetropolisHastings
"""

import LOTlib
from LOTlib import lot_iter
from MetropolisHastings import mh_sample

from copy import copy
from random import randint, random
from math import exp
from random import random
from copy import copy

#def temperature_ladder(min=1.0,max=1.5,steps=5,log=True)


# This staircases, so that each temperature is swapped with the one directly above
# skipping skips within each level!
def parallel_tempering_sample(make_h0, data, steps=9999999999, proposer=None, within_steps=10, temperatures=(1.0,1.1,1.2,1.3,1.4), swaps=1, yield_all=False):
	"""
		If we yield all, we yield everything from every step of every chain
	"""
	
	# a bunch of hypotheses, one for each temperature
	samples = [make_h0() for _ in xrange(len(temperatures))]
	
	for _ in lot_iter(xrange(steps/(within_steps * len(temperatures)))):
		
		for i in xrange(len(temperatures)):
			
			# update this sample
			for s in mh_sample(samples[i], data, within_steps, proposer=proposer, skip=0, likelihood_temperature=temperatures[i]):
				if yield_all: 
					yield s
					
				samples[i] = s # should only be called once, since we skip within step 
		
		for _ in xrange(swaps):
			frm = randint(0, len(temperatures)-2)
			
			# get the joint probability -- since temperature is only on the likelihood, everything cancels
			r = (samples[frm].likelihood) / temperatures[frm+1] + (samples[frm+1].likelihood) / temperatures[frm] - (samples[frm].likelihood) / temperatures[frm] - (samples[frm+1].likelihood) / temperatures[frm+1]
			
			if r>0 or random() < exp(r):
				samples[frm], samples[frm+1] = samples[frm+1], samples[frm]
		
		if not yield_all:
			yield samples[0] # give from the lowest chain
		
if __name__ == "__main__":
	from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
	data = generate_data(300)
	
	make_h0 = lambda : NumberExpression(grammar)

	for h in parallel_tempering_sample(make_h0, data, steps=9999999999):
		print h.posterior_score, h
		