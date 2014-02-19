# -*- coding: utf-8 -*-

"""
	An experimental adaptive method, that runs a bunch of standard mcmc in order to find tree parts that tend to be used in 
	good grammars.
	
	These parts can be added to the PCFG for preferential generations in the PCFG

"""


from collections import defaultdict
import itertools

from copy import copy

import numpy
import scipy
import scipy.optimize

from numpy import log, exp, sqrt, sum
import LOTlib
from LOTlib.Miscellaneous import *

# # # # # # # # # # # # # # # # # # # # # # # # #
# Two quick helper functions

def count_identical_subtrees(t,x):
	"""
		in x, how many are identical to t?
	"""
	return sum([tt==t for tt in x])

def count_identical_nonterminals(t,x):
	""" How many nonterminals in x are of type t? """
	
	# Here we add up how many nodes have the same return type
	# OR how many leaves (from partial trees) have the same returntype
	#print ">>", [k for k in x.all_leaves()]
	
	return sum([tt.returntype==t for tt in x]) +\
	       sum([tt==t for tt in x.all_leaves()])

def count_subtree_matches(t, x):
	return sum(map(lambda tt: tt.partial_subtree_root_match(t), x))
	
# # # # # # # # # # # # # # # # # # # # # # # # #

def print_subtree_adaptations(hypotheses, posteriors, p=0.5, subtree_multiplier=100):
	"""
		Determine how useful it would be to explicitly define each subtree in H across
		all of the (corresponding) posteriors. Here, posteriors is a list of posterior log 
		scores, and we optimize over all of them jointly, seeing how we can change KL
		by altering the prior
		
		We treat hyps as a fixed finite hypothesis space, and assume every subtree considered
		is *not* derived compositionally (although thi scould change in future variants)
	"""
	
	# compute the normalized posteriors
	Ps = map(lognormalize, posteriors)
	
	# Build up a set of unique subtrees by sampling each subtree_multiplier 
	# times its number of nodes
	subtrees = set()
	for h in hypotheses:
		if LOTlib.SIG_INTERRUPTED: break
		for x in h.value: # for each subtree
			for i in xrange(subtree_multiplier):  #take subtree_multiplier random partial subtrees
				subtrees.add( x.random_partial_subtree(p=p) )
	print "# Generated", len(subtrees), "subtrees"
	
	## Now process each, starting with the most simple
	for t in sorted(subtrees, key=lambda t: t.log_probability(), reverse=True):
		if LOTlib.SIG_INTERRUPTED: break
		#print "SUBTREE:", t
		
		# Get some stats on t:
		tlp = t.log_probability()
		tnt = count_identical_nonterminals( t.returntype, t) # How many times is this nonterminal used?
		
		# How many matches of t are there in each H?
		m = numpy.array([ count_subtree_matches(t, h.value) for h in hypotheses])
		assert max(m)>=1
		
		# How many times is the nonterminal used, NOT counting the first of t?
		nt = numpy.array([ count_identical_nonterminals( t.returntype, h.value) for h in hypotheses]) - (tnt-1)*m
		assert min(nt)>=0
		
		# And the prior *not* counting t
		q = lognormalize(numpy.array([ h.value.log_probability() for h in hypotheses]) - tlp * m)
		
		# The function to optimize
		def fnc(p):
			if p <= 0. or p >= 1.: return float("inf") # enforce bounds
		
			newprior = lognormalize( q + log(p)*m + log(1.-p)*nt )
				
			kl = 0.0
			for P in Ps:
				kl += sum( numpy.exp(newprior) * (newprior-P) )
			
			return kl
		

		### TODO: This optimization should be analytically tractable...
		o = scipy.optimize.fmin(fnc, numpy.array([0.1]), xtol=0.0001, ftol=0.0001, disp=0)
		
		print fnc(o[0]), o[0], log(o[0]), t.log_probability(), qq(t)


if __name__ == '__main__':
	
	print "For an example, see Number.OptimalAdapt"
