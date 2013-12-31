from Hypothesis import Hypothesis
from copy import copy, deepcopy
import numpy

class VectorHypothesis(Hypothesis):
	"""
		Store N-dimensional vectors (defaultly with Gaussian proposals)
	"""
	
	def __init__(self, v=None, N=1, proposal=numpy.eye(1)):
		#print numpy.array([0.0]*N), [prposal
		if v is None: v = numpy.random.multivariate_normal(numpy.array([0.0]*N), proposal)
		Hypothesis.__init__(self, v=v)
		
		self.__dict__.update(locals())
		
	def propose(self):
		## NOTE: Does not copy proposal
		newv = numpy.random.multivariate_normal(self.v, self.proposal)
		
		return VectorHypothesis(v=newv, N=self.N, proposal=self.proposal), 0.0 # symmetric proposals
