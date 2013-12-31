import numpy
from copy import copy, deepcopy
import LOTlib
from LOTlib.Miscellaneous import *

class Hypothesis(object):
	"""
		A hypothesis is...
		
		- optionally, compute_likelihood stores self.stored_likelihood, giving the undecayed likelihood on each data point
	"""
	
	def __init__(self, v=None):
		self.set_value(v) # to zero out prior, likelhood, lp
		self.prior, self.likelihood, self.lp = [-Infinity, -Infinity, -Infinity] 
		self.stored_likelihood = None
		POSTERIOR_CALL_COUNTER = 0
	
	def set_value(self, v): 
		""" Sets the (self.)value of this hypothesis to v"""
		self.value = v
		
	def __copy__(self):
		""" Returns a copy of myself by calling copy() on self.value """
		return Hypothesis(v=self.value.copy())
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# All instances of this must implement these:
	
	def likelihood_decay_function(self, i, N, decay):
		"""
		The weight of the likelihood for the ith point out of N with the given decay parameter.
		Generally, this should be a power law decay
		i - What data point (0-indexed)
		N - how many total data points
		"""
		return (N-i+1)**(-decay)
	
	def compute_prior(self):
		""" computes the prior and stores it in self.prior"""
		print "*** Must implement compute_prior"
		assert False # Must implement this

	
	def compute_likelihood(self, d, decay=0.0):
		""" 
		Compute the likelihood of the *array* d, with the specified likelihood decay
		This also stores the *undecayed* non-culmulative likelihood of each data point in self.stored_likelihood
		"""
		print "*** Must implement compute_likelihood"
		assert False # Must implement this
	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Methods for accessing likelihoods etc. on a big arrays of data
	
	def get_culmulative_likelihoods(self, ll_decay=0.0, shift_right=True):
		"""
		Compute the culmulative likelihoods on the stored data
		This gives the likelihood on the first data point, the first two, first three, etc, appropriately decayed
		using the 'pointwise' likelihoods stored in self.stored_likelihood.
		NOTE: This is O(N^2) (for power law decays; for exponential it could be linear)
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
			if ll_decay==0.0: # shortcut if no decay
				sm = numpy.sum(self.stored_likelihood[0:n])
			else: 
				sm = numpy.sum( [self.stored_likelihood[j] * self.likelihood_decay_function(j, n, ll_decay) for j in xrange(n) ])
			out.append(sm)
		return numpy.array(out)
			
	def get_culmulative_posteriors(self, ll_decay=0.0, shift_right=False):
		"""
		returns the posterior with the i'th stored CULMULATIVE likelihood, using the assumed decay
		"""
		return self.get_culmulative_likelihoods(ll_decay=ll_decay, shift_right=shift_right) + self.prior
	
	def propose(self): 
		""" Generic proposal used by MCMC methods"""
		print "*** Must implement propose"
		assert False # Must implement this
	
	# this updates last_prior and last_likelihood
	def compute_posterior(self, d):
		LOTlib.BasicPrimitives.LOCAL_PRIMITIVE_OPS = 0 # Reset this
		p = self.compute_prior()
		l = self.compute_likelihood(d)
		self.lp = p+l
		return [p,l]
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# optional implementation
	# if you do gibbs sampling you need:
	def enumerative_proposer(self): pass
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	## These are just handy:
	def __str__(self):
		return str(self.value)
	def __repr__(self): return str(self)
		
	# for hashing hypotheses
	def __hash__(self): return hash(self.value)
	def __cmp__(self, x): return cmp(self.value,x)
	
	# this is for heapq algorithm in FiniteSample, which uses <= instead of cmp
	# since python implements a "min heap" we can compar elog probs
	def __le__(self, x): return (self.lp <= x.lp)
	def __eq__(self, other): 
		return (self.value.__eq__(other.value))
	def __ne__(self, other): return (self.value.__ne__(other.value))

