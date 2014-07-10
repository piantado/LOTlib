from copy import copy

import numpy

from LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import Infinity, beta

from collections import defaultdict

def get_rule_counts(grammar, t):
	"""
		A list of vectors of counts of how often each nonterminal is expanded each way
		
		TODO: This is probably not super fast since we use a hash over rule ids, but
			it is simple!
	"""
	
	counts = defaultdict(int) # a count for each hash type
	
	for x in t:
		if x.ruleid >= 0: counts[x.ruleid] += 1
	
	# and convert into a list of vectors (with the right zero counts)
	out = []
	for nt in grammar.rules.keys():
		v = numpy.array([ counts.get(r.rid,0) for r in grammar.rules[nt] ])
		out.append(v)
	return out

def RR_prior(grammar, t, alpha=1.0):
		"""
			Compute the rational rules prior from Goodman et al. 
			
			NOTE: This has not yet been extensively debugged, so use with caution
			
			TODO: Add variable priors (different vectors, etc)
		"""
		lp = 0.0
		
		for c in get_rule_counts(grammar, t):
			theprior = numpy.array( [alpha] * len(c), dtype=float )
			#theprior = np.repeat(alpha,len(c)) # Not implemented in numpypy
			lp += (beta(c+theprior) - beta(theprior))
		return lp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RationalRulesLOTHypothesis(LOTHypothesis):
	"""
		A FunctionHypothesis built from a grammar.
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions.
		
	"""	
	def __init__(self, grammar, rrAlpha=1.0, *args, **kwargs):
		"""
			Everything is passed to LOTHypothesis
		"""
		self.rrAlpha = rrAlpha
		
		LOTHypothesis.__init__(self, grammar, *args, **kwargs)


	def __copy__(self):
		"""
			Return a copy of myself.
		"""

		# Since this is inherited, call the constructor on everything, copying what should be copied
		thecopy = type(self)(self.grammar, rrAlpha=self.rrAlpha) 
		
		# And then then copy the rest
		for k in self.__dict__.keys():
			if k not in ['self', 'grammar', 'value', 'proposal_function', 'f']:
				thecopy.__dict__[k] = copy(self.__dict__[k])
		
		return thecopy

	def compute_prior(self): 
		"""
		
		"""
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			# compute the prior with either RR or not.
			self.prior = RR_prior(self.grammar, self.value, alpha=self.rrAlpha) / self.prior_temperature
			
		self.posterior_score = self.prior + self.likelihood
			
		return self.prior



		
