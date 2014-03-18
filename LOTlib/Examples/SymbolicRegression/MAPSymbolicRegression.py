# -*- coding: utf-8 -*-
"""
	Symbolic regression that fits parameters using a MAP estimate of the continuous parameters.
	We put a simple bayesian prior on these constants, and use it to compute MAPs
"""

from Shared import * # imports G
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
import scipy, numpy
from scipy.optimize import fmin
from numpy.random import normal
from math import sin

NCONSTANTS = 4
CONSTANT_NAMES = ['C%i'%i for i in xrange(NCONSTANTS) ] 
CONSTANT_SD = 1.0

CHAINS = 1
STEPS = 500000
SKIP = 0
LL_SD = 0.1 # the SD of the likelihood
NDATA = 50
MEMOIZE = 1000 # 0 means don't memoize
MAXITER=100 # max iterations for the optimization to run
MAX_INITIALIZE=25 # max number of random numbers to try initializing with

## The target function for symbolic regression 
target = lambda x: 3.*x + sin(4.3/x)

# Supplement the grammar
for c in CONSTANT_NAMES: G.add_rule('EXPR', c, None, 5.0)


class MAPSymbolicRegressionHypothesis(GaussianLOTHypothesis):
	"""
		This is a quick hack to try out symbolic regression with constants just fit. 
		This hacks it by defining a self.CONSTANT_VALUES that are automatically read from 
		get_function_responses (overwritten). We can then change them and repeatedly compute the 
		likelihood to optimize
		
	"""
	
	def value2function(self, value):
		"""
		Overwrite this from FunctionHypothesis. Here, we add args for the constants so we can use them
		"""
		return evaluate_expression(value, args=self.args+CONSTANT_NAMES)
	
	
	def get_function_responses(self, data, *args, **kwargs):
		return GaussianLOTHypothesis.get_function_responses(self, data, *self.CONSTANT_VALUES) # Pass in the constant names
	
	def compute_likelihood(self, data):
		
		def llgivenC(fit_params):
			self.CONSTANT_VALUES = fit_params.tolist() # set these
			# And return the original likelihood, which by get_function_responses above uses this
			constant_prior = sum(map(lambda x: normlogpdf(x,0.0,CONSTANT_SD), self.CONSTANT_VALUES))
			return -(GaussianLOTHypothesis.compute_likelihood(self, data) + constant_prior)

		for init in xrange(MAX_INITIALIZE):
			p0 = normal(0.0, CONSTANT_SD, NCONSTANTS)
			res = fmin(llgivenC, p0, disp=False, maxiter=MAXITER)
			if llgivenC(res) < Infinity: break
		
		self.CONSTANT_VALUES = res
		
		if llgivenC(res) < Infinity:  self.likelihood = -llgivenC(res) ## must invert since it's a negative
		else:                         self.likelihood = -Infinity
			
		self.posterior_score = self.prior + self.likelihood
		return self.likelihood


# Make up some learning data for the symbolic regression
def generate_data(data_size):
	
	# initialize the data
	data = []
	for i in range(data_size): 
		x = random()
		y = target(x) + normal()*LL_SD
		data.append( FunctionData(input=[x], output=y, ll_sd=LL_SD) )
	
	return data
	

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# starting hypothesis -- here this generates at random

data = generate_data(NDATA) # generate some data
h0 = MAPSymbolicRegressionHypothesis(G)
h0.CONSTANT_VALUES = numpy.zeros(NCONSTANTS)

for h in mh_sample(h0, data, STEPS, skip=SKIP, trace=F, debug=F, memoize=MEMOIZE):
	print h.posterior_score, h.likelihood, h.prior, h.CONSTANT_VALUES, qq(h)
	
