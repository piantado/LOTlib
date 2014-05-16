# -*- coding: utf-8 -*-

"""
	proposer - should return a *new copy* of a hypothesis, and a forward-backward probability
	
	TODO: Put the mh ratio in a separate function, that correctly handles -inf and nan
	
	## TODO: Add simulated_tempering -- where you move between transition. Then you 
	# can count how often you are at each temperature and adjust easily
	#See Geyer & Thompson 1994 - http://www.stat.tamu.edu/~fliang/Geyer.pdf

"""

import time
from random import random
from math import log, exp, isnan

import LOTlib
from LOTlib import lot_iter
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Memoization import BoundedDictionary
from MHShared import *

def mh_sample(current_sample, data, steps=1000000, proposer=None, skip=0, prior_temperature=1.0, ll_temperature=1.0, acceptance_temperature=1.0, trace=False, debug=False, stats=None, memoize=0):
	"""
		current_sample - the starting hypothesis
		data - the conditioning data
		steps - how many steps to run
		proposer - if not None, use this instead of inh.propose() to compute proposals
		skip - only return samples every this many steps
		trace - if true, we display the random number, proposal, current hypothesis, and sample proposal
		stats - if not none, then we store sampling information in it (hopefully, a MHStats object)
		memoize - if > 0, we memoize this many
	"""
	
	if memoize > 0:
		mem = BoundedDictionary(N=memoize)
		
	for mhi in lot_iter(xrange(steps)):
	
		for skp in lot_iter(xrange(skip+1)):
		
			if proposer is None: p, fb = current_sample.propose()
			else:                p, fb = proposer(current_sample)
			
			# A speed to check to see if identical -- not much faster
			#if fb==0.0 and p == current_sample:
				#continue # next iteration of the skip loop
			
			# either compute this, or use the memoized version
			if memoize > 0:
				if p in mem:
					np, nl = mem[p]
					p.posterior_score = (np+nl) # update this since it won't be set
				else: 
					np, nl = p.compute_posterior(data)
					mem[p] = (np,nl)
			else:
				np, nl = p.compute_posterior(data)
			
			#print np, nl, current_sample.prior, current_sample.likelihood
			prop = (np/prior_temperature+nl/ll_temperature)
			cur  = (current_sample.prior/prior_temperature + current_sample.likelihood/ll_temperature)
			
			if debug: 
				print "# Proposing: ", prop, cur, fb
				print "# From: ", current_sample
				print "# To:   ", p
			
			if MH_acceptance(cur, prop, fb, acceptance_temperature=acceptance_temperature):
				current_sample = p
				
				if stats is not None: stats['accept'] += 1
				if debug: print "# Accept!"
			else:
				if debug: print "# Reject."
			
			if stats is not None: stats['total'] += 1
			
			if debug: print "\n\n";
		
		if trace: 
			print current_sample.posterior_score, current_sample.likelihood, current_sample.prior, qq(current_sample)
		
		yield current_sample
		
		

	#print mem.hits, mem.misses
			
# this does out special mix of mh and gibbs steps
def mhgibbs_sample(inh, data, steps, proposer=None, mh_steps=10, gibbs_steps=10, skip=0, temperature=1.0):
	current_sample = inh
	for mhi in lot_iter(xrange(steps)):
		for skp in xrange(skip+1):
			for k in mh_sample(current_sample, data, 1, proposer=proposer, skip=mh_steps, temperature=temperature): current_sample = k
			for k in gibbs_sample(current_sample, data, 1, skip=gibbs_steps, temperature=temperature): current_sample = k
		yield current_sample

		
	
if __name__ == "__main__":
	
	from LOTlib.Examples.Number.Shared import *
	
	data = generate_data(500)
	h0 = NumberExpression(G)	
	for h in mh_sample(h0, data, 10000):
		print q(get_knower_pattern(h)), h.lp, h.prior, h.likelihood, q(h)
		  
	

