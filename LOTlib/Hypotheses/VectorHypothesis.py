from Hypothesis import Hypothesis
from copy import copy, deepcopy
import numpy

class VectorHypothesis(Hypothesis):
	"""
		Store N-dimensional vectors (defaultly with Gaussian proposals)
	"""
	
	def __init__(self,value=None, N=1, proposal=numpy.eye(1)):
		#print numpy.array([0.0]*N), [prposal
		if value is None: value = numpy.random.multivariate_normal(numpy.array([0.0]*N), proposal)
		Hypothesis.__init__(self, value=value)
		
		self.__dict__.update(locals())
		
	def propose(self):
		## NOTE: Does not copy proposal
		newv = numpy.random.multivariate_normal(self.value, self.proposal)
		
		return VectorHypothesis(value=newv, N=self.N, proposal=self.proposal), 0.0 # symmetric proposals
