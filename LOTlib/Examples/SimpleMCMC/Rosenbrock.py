
"""

	Just playing around with vector-valued hypotheses. This is a simple sampler for a posterior shaped like
	the exp(-RosenbrockFunction)
"""

from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import *
import numpy

class RosenbrockSampler(VectorHypothesis):
	
	def __init__(self, value=None):
		if value is None: value = numpy.array([0.0, 0.0])
		VectorHypothesis.__init__(self, value=value, N=2, proposal=numpy.eye(2)*0.1)
		
	"""
		MCMC plays nicest if we have defined prior and likelihood, and just don't touch compute_posterior
	"""
	def compute_likelihood(self, data): 
		self.likelihood = 0.0
		self.posterior_score = self.prior + self.likelihood
		return self.likelihood
		
	def compute_prior(self):
		x,y = self.value
		self.prior = -((1.0-x)**2.0 + 100.0*(y-x**2.0)**2.0)
		self.posterior_score = self.prior + self.likelihood
		return self.prior
		
	
	def propose(self):
		## NOTE: Does not copy proposal
		newv = numpy.random.multivariate_normal(self.value, self.proposal)
		return RosenbrockSampler(value=newv), 0.0 # from symmetric proposals
		
if __name__ == "__main__":
	
	N = 1
	initial_hyp = RosenbrockSampler()
	
	for x in mh_sample(initial_hyp, [], 1000000, skip=100, trace=False): 
		print x, x.posterior_score