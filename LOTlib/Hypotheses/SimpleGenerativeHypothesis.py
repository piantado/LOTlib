from LOTHypothesis import LOTHypothesis
from copy import copy, deepcopy
from collections import defaultdict
from math import log, exp


class SimpleGenerativeHypothesis(LOTHypothesis):
	"""
		Here, the data is a hash from strings to counts. 
		Each function eval results in a string, and the likelihood is the probability of generating the 
		observed data. The function eval results in a thunk
		
		NOTE: FOR NOW, Insert/Delete moves are taken off because of some weirdness with the lambda thunks
	"""
	def __init__(self, G, nsamples=100, **kwargs): 
		""" kwargs should include ll_sd """
		assert kwargs.get('args', None) is None, "Cannot specify args to SimpleGenerativeHypothesis"
		self.nsamples=nsamples
		
		LOTHypothesis.__init__(self, G, args=[], **kwargs) # this is simple-generative since args=[] (a thunk)
	
	def compute_single_likelihood(self, datum):
		assert False, "Should not call this!"
	
	def compute_likelihood(self, data, sm=0.001):
		"""
			sm smoothing counts are added to existing bins of counts (just to prevent badness)
		"""
		#print self
		assert len(self.args) == 0, "Can only use SimpleGenerativeHypothesis on thunks!"
		assert isinstance(data, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"
		
		self.llcounts = defaultdict(int)
		for i in xrange(self.nsamples):
			self.llcounts[ self() ] += 1
		
		self.likelihood = sum([ data[k] * (log(self.llcounts[k] + sm)-log(self.nsamples + sm*len(data.keys())) ) for k in data.keys() ])
		
		self.posterior_score = self.likelihood + self.prior
		
		return self.likelihood
		
		