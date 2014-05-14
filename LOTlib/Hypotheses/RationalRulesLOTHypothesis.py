from copy import copy

from LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import Infinity

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
			self.prior = self.grammar.RR_prior(self.value, alpha=self.rrAlpha) / self.prior_temperature
			
		self.posterior_score = self.prior + self.likelihood
			
		return self.prior
