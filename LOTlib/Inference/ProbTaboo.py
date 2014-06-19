"""
	The more times we visit a hypothesis, the more we decrease its prior
	
	TODO: Try version where penalty decreases with time!
	TODO: This currently only extends LOTHypotheses, since we have to handle casting
	      inside of h0 to WrapperClass. HOWEVER, we could make WrapperClass just dispatch the right methods 
	      if they don't exist
"""


from collections import defaultdict

import LOTlib
from LOTlib import lot_iter
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from copy import copy

def ptaboo_search(h0, data, steps, skip=0, noisy_memoize=1000, seen_penalty=1.0):

	seen_count = defaultdict(int)
	
	# define a wrapper class that overwrites prior with our penalized version
	class WrapperClass(type(h0)):
	
		def compute_prior(self):
	
			self.rawprior =  type(h0).compute_prior(self) # save the prior for use if we want to convert back
			self.prior = self.rawprior - seen_count[self]*seen_penalty
			self.lp = self.prior + self.likelihood
			return self.prior
		
		def fixlp(self):
			"""
				Temporarily fix our log probability returned
			"""
			self.prior = self.rawprior
			self.lp = self.prior + self.likelihood
	
	myh0 = WrapperClass(h0.grammar, v=h0.value) ## TODO: NOTE HERE WE ASSUME grammar IS TAKEN!
	
	# Now just run standard MCMC:
	for h in lot_iter(mh_sample(myh0, data, steps, skip=skip)):
		# THIS IS VERY BIZARRE: 
		# We don't yield a copy, so we fixlp, yield, and then re-compute the prior to restore the lp
		# to the current sample
		#h.fixlp()
		#yield h
		#h.compute_prior() # 
		
		# Slower way to do it, just copy the value
		h0.set_value(h.value)
		h0.compute_posterior(data)
		yield h0
		
		seen_count[h] += 1


if __name__ == "__main__":
	
	from LOTlib.Examples.Number.Shared import *
	
	data = generate_data(500)
	h0 = NumberExpression(grammar)	
	for h in ptaboo_search(h0, data, 10000):
		#h.revert() # undoes the craziness with the prior
		print q(get_knower_pattern(h)), h.lp, h.prior, h.likelihood, q(h)
		
		
	
		