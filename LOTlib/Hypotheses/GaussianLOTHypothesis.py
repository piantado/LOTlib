from LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import *
from copy import copy, deepcopy

class GaussianLOTHypothesis(LOTHypothesis):
	"""
		Like LOTHypothesis but has a Gaussian likelihood
	"""
	
	def __init__(self, G, value=None, f=None, prior_temperature=1.0, ll_decay=0.0, proposal_function=None): 
		self.__dict__.update(locals()) # must come first or else proposal_function is overwritten
		LOTHypothesis.__init__(self, G, value=value, f=f, proposal_function=proposal_function)
		
		
	def compute_likelihood(self, data):
		""" Compute the likelihood with a Gaussian"""
		
		# set up to store this data
		self.stored_likelihood = [None] * len(data)
		
		# compute responses to all data
		responses = self.get_function_responses(data) # get my response to each object
		
		N = len(data)
		self.likelihood = 0.0
		for i in xrange(N):
			self.stored_likelihood[i] = normlogpdf(responses[i], data[i].output, data[i].ll_sd)
			
			# the total culmulative decayed likeliood
			self.likelihood += self.stored_likelihood[i] * self.likelihood_decay_function(i, N, self.ll_decay)
		
		self.posterior_score = self.prior + self.likelihood
		
		return self.likelihood