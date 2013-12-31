from LOTHypothesis import LOTHypothesis
from copy import copy, deepcopy

class SimpleGenerativeHypothesis(LOTHypothesis):
	"""
		Here, the data is a hash from strings to counts. 
		Each function eval results in a string, and the likelihood is the probability of generating the 
		observed data. The function eval results in a thunk
		
		NOTE: FOR NOW, Insert/Delete moves are taken off because of some weirdness with the lambda thunks
	"""
	def __init__(self, G): 
		""" kwargs should include ll_sd """
		LOTHypothesis.__init__(self, G, args=[]) # this is simple-generative since args=[] (a thunk)
		self.__dict__.update(locals())
	
	def compute_likelihood(self, data, nsamples=250, sm=1e-3):
		"""
			sm smoothing counts are added to existing bins of counts (just to prevent badness)
		"""
		#print self
		assert len(self.args) == 0, "Can only use SimpleGenerativeHypothesis on thunks!"
		assert isinstance(data, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"
		
		self.llcounts = defaultdict(int)
		for i in xrange(nsamples):
			self.llcounts[ self() ] += 1
		
		self.likelihood = sum([ data[k] * (log(self.llcounts[k] + sm)-log(nsamples + sm*len(data.keys())) ) for k in data.keys() ])
		
		return self.likelihood
		
		