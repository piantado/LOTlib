from LOTHypothesis import LOTHypothesis
class RationalRulesLOTHypothesis(LOTHypothesis):
	"""
		A FunctionHypothesis built from a grammar.
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions.
		
	"""	
	
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
