# -*- coding: utf-8 -*-

"""
	An experimental adaptive method, that runs a bunch of standard mcmc in order to find tree parts that tend to be used in 
	good grammars.
	
	These parts can be added to the PCFG for preferential generations in the PCFG

"""


from collections import defaultdict
import itertools

import numpy
import scipy
import scipy.optimize

from numpy import log, exp, sqrt, sum

from LOTlib.Miscellaneous import *

# # # # # # # # # # # # # # # # # # # # # # # # #
# Two quick helper functions

def count_identical_subtrees(t,x):
	"""
		in x, how many are identical to t?
	"""
	cnt = 0
	for tt in x:
		if tt == t: cnt += 1
	return cnt

def count_identical_nonterminals(t,x):
	""" How many nonterminals in x are of type t? """
	cnt = 0
	for tt in x:
		if tt.returntype == t: cnt += 1
	return cnt
	
# # # # # # # # # # # # # # # # # # # # # # # # #

def print_subtree_adaptations(hyps):
	"""
		Determine how useful it would be to explicitly define each subtre ein H.
		We assume we can set its probability freely, and see how closely we can move
		the prior towards the posterior, as measured by KL.
		
		We treat hyps as a fixed finite hypothesis space, and assume every subtree considered
		is *not* derived compositionally (although thi scould change in future variants)
	"""
	
	# a fixed order, tossing the -inf hypotheses that might come our way
	H = filter( lambda h: h.posterior_score > -Infinity, hyps)
	
	# Determine their normalized probability
	P = lognormalize( numpy.array( [h.posterior_score for h in H]) )

	# all subtrees in all hypotheses
	subtrees = set(itertools.chain(*[h.value for h in H]))
	
	for t in sorted(subtrees, key=lambda t:t.log_probability(), reverse=True):
		
		tlp = t.log_probability()
		tnt = count_identical_nonterminals( t.returntype, t) # don't count these in K

		# How many identical trees are in each H?
		i = numpy.array([ count_identical_subtrees(t, h.value) for h in H ])
		
		# How many times is the nontemrinal used, NOT counting the first of t?
		n = numpy.array([ count_identical_nonterminals( t.returntype, h.value) for h in H]) - (tnt-1)*i

		# And the prior *not* counting t
		q = lognormalize(numpy.array([ h.value.log_probability() for h in H]) - tlp * i)
		
		def fnc(p):
			if p <= 0. or p >= 1.: return float("inf") # enforce bounds

			newq = lognormalize( q + log(p)*i + log(1.-p)*n )
			
			kl = sum( numpy.exp(newq) * (newq-P) )
			
			return kl
		
		"""
			TODO: This optimization should be analytically tractable...
			#a =  log(sum(exp(P)*n)) - log(sum(exp(P)*i))
		"""		
		o = scipy.optimize.fmin(fnc, numpy.array([0.1]), xtol=0.0001, ftol=0.0001, disp=0)
		
		print fnc(o[0]), o[0], log(o[0]), t.log_probability(), qq(t)


if __name__ == '__main__':
	
	print "For an example, see Number.OptimalAdapt"
