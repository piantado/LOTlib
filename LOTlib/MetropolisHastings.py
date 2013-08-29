# -*- coding: utf-8 -*-

"""
	proposer - should return a *new copy* of a hypothesis, and a forward-backward probability
	
	TODO: Put the mh ratio in a separate function, that correctly handles -inf and nan
	
	## TODO: Add simulated_tempering -- where you move between transition. Then you 
	# can count how often you are at each temperature and adjust easily
	#See Geyer & Thompson 1994 - http://www.stat.tamu.edu/~fliang/Geyer.pdf

"""
from random import random, randint, shuffle
from math import log, exp, isnan
from copy import copy, deepcopy

import time
import LOTlib
import LOTlib.Hypothesis 

from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import FiniteBestSet
from Memoization import *
from collections import defaultdict

class MHStats(defaultdict):
	def __init__(self):
		defaultdict.__init__(self, int)
	
	def acceptance_ratio(self):
		if self.get('total') > 0:
			return float(self.get('accept',0)) / float(self.get('total',1))
		else:   return None


def mh_sample(current_sample, data, steps=float("inf"), proposer=None, skip=0, temperature=1.0, trace=False, stats=None):
	"""
		current_sample - the starting hypothesis
		data - the conditioning data
		steps - how many steps to run
		proposer - if not None, use this instead of inh.propose() to compute proposals
		skip - only return samples every this many steps
		temperature - the sampler temperature -- only on the likelihood (TODO: FIX)
		trace - if true, we display the random number, proposal, current hypothesis, and sample proposal
		stats - if not none, then we store sampling information in it (hopefully, a MHStats object)
	"""
	
	mhi = 0
	while mhi < steps:
		for skp in xrange(skip+1):
			
			if proposer is None: p, fb = current_sample.propose()
			else:                p, fb = proposer(current_sample)
				
			np, nl = p.compute_posterior(data)
			
			#print np, nl, current_sample.prior, current_sample.likelihood
			prop = (np+nl/temperature)
			cur  = (current_sample.prior + current_sample.likelihood/temperature)
			if math.isnan(cur) or (cur==-inf and prop==-inf): # if we get infs or are in a stupid state, let's just sample from the prior so things don't get crazy
				r = -log(2.0) #  just choose at random -- we can't sample priors since they may be -inf both
			elif math.isnan(prop): #never accept
				r = float("-inf")
			else:
				r = prop-cur-fb
			
			if trace: 
				print "# Proposing: ", r, prop, cur, fb
				print "# From: ", current_sample
				print "# To:   ", p
			
			if r >= 0.0 or random() < exp(r):
				current_sample = p
				
				if stats is not None: stats['accept'] += 1
				if trace: print "# Accept!"
			else:
				if trace: print "# Reject."
			
			if stats is not None: stats['total'] += 1
			
			if trace: print "\n\n";
			
		yield current_sample
		
		if LOTlib.SIG_INTERRUPTED: break
		mhi += 1
			
# this does out special mix of mh and gibbs steps
def mhgibbs_sample(inh, data, steps, proposer=None, mh_steps=10, gibbs_steps=10, skip=0, temperature=1.0):
	current_sample = inh
	for mhi in xrange(steps):
		for skp in xrange(skip+1):
			for k in mh_sample(current_sample, data, 1, proposer=proposer, skip=mh_steps, temperature=temperature): current_sample = k
			for k in gibbs_sample(current_sample, data, 1, skip=gibbs_steps, temperature=temperature): current_sample = k
		yield current_sample
		if LOTlib.SIG_INTERRUPTED: break

		
## TODO: CHECK THIS--STILL NOT SURE THIS IS RIGHT
# a helper function for temperature transitions -- one single MH step, returning a new sample
# this allows diff. temps for top and bottom
def tt_helper(xi, data, tnew, told, proposer):
	if proposer is None: xinew, fb = xi.propose()
	else:                xinew, fb = propose(xi)
	xinew.compute_posterior(data)
	r = xinew.prior + xinew.likelihood / tnew - (xi.prior + xi.likelihood / told) - fb
	if r > 0.0 or random() < exp(r):
		return xinew 
	else:   return xi
	
	
## TODO: DEBUG THIS -- especially for asmmetric proposals.. ALSO TO BE SURE IT IS RIGHT
def tempered_transitions_sample(inh, data, steps, proposer=None, skip=0, temperatures=[1.0, 1.05, 1.1], stats=None):
	current_sample = inh
	
	LT = len(temperatures)
	
	for mhi in xrange(steps):
		for skp in xrange(skip+1):
		
			xi = current_sample # do not need to copy this
			totlp = 0.0 #(xi.lp / temperatures[1]) - (xi.lp / temperatures[0])
			
			for i in xrange(0,LT-2): # go up
				xi = tt_helper(xi, data, temperatures[i+1], temperatures[i], proposer)
				totlp = totlp + (xi.prior + xi.likelihood / temperatures[i+1]) - (xi.prior + xi.likelihood / temperatures[i])
			
			# do the top:
			xi = tt_helper(xi, data, temperatures[LT-1], temperatures[LT-1], proposer) 
			
			for i in xrange(len(temperatures)-2, 0, -1): # go down
				xi = tt_helper(xi, data, temperatures[i], temperatures[i], proposer) 
				totlp = totlp + (xi.prior + xi.likelihood / temperatures[i]) - (xi.prior + xi.likelihood / temperatures[i+1])
			
			if random() < exp(totlp): current_sample = xi # copy this over

		yield current_sample
		if LOTlib.SIG_INTERRUPTED: break



# This staircases, so that each temperature is swapped with the one directly above
# skipping skips within each level!
def parallel_tempering_sample(inh, data, steps, proposer=None, within_steps=10, temperatures=(1.0, 1.5, 2.0, 3.0, 4.0), swaps=1):
	
	# a bunch of hypotheses, one for each temperature
	samples = map(deepcopy, [ inh ] * len(temperatures) ) ##TODO: Maybe deepcopy is not what we want here?
	
	for mhi in xrange(steps):
		
		for i in xrange(len(temperatures)):
			
			# update this sample
			for s in mh_sample(samples[i], data, 1, proposer=proposer, skip=within_steps, temperature=temperatures[i]):
				samples[i] = s # should only be called once, since we skip within step 
		
		for sw in xrange(swaps):
			frm = randint(0, len(temperatures)-2)
			
			# get the joint probability -- since temperature is only on the likelihood, everything cancels
			r = (samples[frm].likelihood / temperatures[frm+1] + samples[frm+1].likelihood / temperatures[frm]) - (samples[frm].likelihood / temperatures[frm] + samples[frm+1].likelihood / temperatures[frm+1])
			
			if random() < exp(r):
				tmp = samples[frm]
				samples[frm] = samples[frm+1]
				samples[frm+1] = tmp
		
		yield samples[0] # give from the lowest chain
		if LOTlib.SIG_INTERRUPTED: break

		
def mh_swappy_chains(inh, data, steps, skip=0, temperature=1.0, chains=10, mix=0.5):
	"""
		Run chains chains and mix with subtree swaps between (random) chains	
		
		Swapping is nice here because in the limit of chains, we have that many 
		
		TODO: Implement
	"""
	
	pass
	

def gibbs_sample(inh, data, steps, skip=0, temperature=1.0, dimensions=None, randomize_dimensions=True):
	"""
		 This uses inh.enumerative_proposer() to yield a bunch of different proposed values
		 and then gibbs samples from them.
		 
		 dimensions - if not None, we loop over the element of dimensions passing them as an 
		 argument to enumerative_proposer
		 
		 randomize_dimensions - do we shuffle the order of dimensions on each loop?
		 
	"""
	
	current_sample = inh
	
	# the dimension loop
	if dimensions is not None:
		mydims = [ [x] for x in dimensions ] # wrap each element in a list, to pass
		if randomize_dimensions: shuffle(mydims)
	else: # if no dimesion arguments, provide no arguments to enumerative_proposer
		mydims = [ [] ]
	dimidx = 0
		
	for gi in xrange(steps):
		for skp in xrange(skip+1):
			if LOTlib.SIG_INTERRUPTED: return
			
			samples = [] # this is just used to store each gibbs value
			
			# for every proposal that is given back
			for p in current_sample.enumerative_proposer(*mydims[dimidx]):
				p.compute_posterior(data)
				p.lp = p.prior + p.likelihood / temperature # scale here for temperature (TODO: This differs from all others!)
				samples.append(p)
				#print p.lp, q(p.word_idx[0]), q(p.word_idx[1])
			
			# choose from the "samples" to get the sample--this works because it defines "lp"
			current_sample = weighted_sample(samples)
			#print [x.lp for x in samples]
			#print current_sample.lp, current_sample
			
			#print "\t", current_sample
			yield current_sample
			if LOTlib.SIG_INTERRUPTED: break
			
			# manage the dimension loop
			#print "==>", dimidx, mydims[dimidx]
			if dimensions is not None:
				dimidx += 1 
				if dimidx >= len(mydims)-1:
					dimidx = 0
					if randomize_dimensions: shuffle(mydims)

