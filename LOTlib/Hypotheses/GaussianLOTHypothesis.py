from LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import *

class GaussianLOTHypothesis(LOTHypothesis):
	"""
		Like LOTHypothesis but has a Gaussian likelihood
	"""
	
	def __init__(self, grammar, value=None, f=None, prior_temperature=1.0, proposal_function=None, **kwargs): 
		self.__dict__.update(locals()) # must come first or else proposal_function is overwritten
		LOTHypothesis.__init__(self, grammar, value=value, f=f, proposal_function=proposal_function, **kwargs)
		
		
	def compute_single_likelihood(self, datum):
		""" Compute the likelihood with a Gaussian"""
		return normlogpdf(self(*datum.input), datum.output, datum.ll_sd)
