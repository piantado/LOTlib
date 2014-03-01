import LOTlib
from LOTlib.Hypotheses.Hypothesis import Hypothesis
import numpy

class DecayedLikelihoodHypothesis(Hypothesis):
	"""
		This implements a likelihood decay such that more recent data
		is weighted more strongly, via the parameter ll_decay
		
		By default, we store the likelihoods for each data point (as we may fit ll_decay)
		
		TODO: MAKE THIS WORK AGAIN! THIS HAS BEEN UPDATED AFTER EXTRACTING FROM HYPOTHESIS CLASS
	"""
	
	def __init__(self, value=None, ll_decay=0.0, **kwargs):
		Hypothesis.__init__(self, value=value, **kwargs)
		self.ll_decay = ll_decay # store this
		self.stored_likelihood = None
		
		
	def likelihood_decay_function(self, i, N, decay):
		"""
		The weight of the likelihood for the ith point out of N with the given decay parameter.
		Generally, this should be a power law decay
		i - What data point (0-indexed)
		N - how many total data points
		"""
		return (N-i+1)**(-decay) 
	

	def get_culmulative_likelihoods(self, shift_right=True):
		"""
		Compute the culmulative likelihoods on the stored data
		This gives the likelihood on the first data point, the first two, first three, etc, appropriately decayed
		using the 'pointwise' likelihoods stored in self.stored_likelihood.
		NOTE: This is O(N^2) (for power law decays; for geometric it could be linear)
		returns: a numpy array of the likelihoods
		
		- decay - the memory decay
		- shift_right -- do we insert a "0" at the beginning (corresponding to inferences with 0 data), and then delete one from the end?
			       - So if you do posterior predictives, you want shift_right=True
		"""
		assert self.stored_likelihood is not None
		
		offset = 0
		if shift_right: offset = 1
		
		out = []
		for n in xrange(1-offset,len(self.stored_likelihood)+1-offset):
			if self.ll_decay==0.0: # shortcut if no decay
				sm = numpy.sum(self.stored_likelihood[0:n])
			else: 
				sm = numpy.sum( [self.stored_likelihood[j] * self.likelihood_decay_function(j, n, self.ll_decay) for j in xrange(n) ])
			out.append(sm)
		return numpy.array(out)
			
			
	def get_culmulative_posteriors(self, shift_right=False):
		"""
		returns the posterior with the i'th stored CULMULATIVE likelihood, using the assumed decay
		"""
		return self.get_culmulative_likelihoods(shift_right=shift_right) + self.prior
	
	
	def compute_likelihood(self, data):
		"""
			This is overwritten, writes to stored_likelihood, and thenc alls get_culmulative_likelihoods
		"""
		self.stored_likelihood = map( self.compute_single_likelihood, data)
		culm_lls = self.get_culmulative_likelihoods(self) # TODO: We don't actually need the entire array--shouldn't pass around!
		
		self.likelihood = culm_lls[-1]/self.likelihood_temperature
		
		
		self.posterior_score = self.prior + self.likelihood
		
		return self.likelihood