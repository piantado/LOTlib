# -*- coding: utf-8 -*-

"""
	The main class for samping. This computes log probabilities and proposals
	
	
	NOTE: In implementation, self.prior, self.likelihood and self.lp must be set whenever prior, likelihood are computed
	
		
	TODO:
		- Probably should collapse FunctionHypothesis and STandardHypothesis toegether,since having a FunctionHypothesis is almost always a bad idea (no prior)
		- type(self).__new__(...args...) -- should be able to use this in the copy constructor, instead of having to compulsively define .copy()
		- Redo these with a "params" dictionary for each hypothesis, which stores things like decay, alpha, etc. 
			And then you can just copy this over with the copy constructor
"""
from LOTlib.Miscellaneous import *
import LOTlib.BasicPrimitives # needed to eval __call__ here, since that's where they are bound
from LOTlib.DataAndObjects import FunctionData,UtteranceData
from copy import copy
import numpy
import sys


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Hypothesis(object):
	"""
		A hypothesis is...
		
		- optionally, compute_likelihood stores self.stored_likelihood, giving the undecayed likelihood on each data point
	"""
	
	def __init__(self, v=None):
		self.set_value(v) # to zero out prior, likelhood, lp
		self.prior, self.likelihood, self.lp = [-Infinity, -Infinity, -Infinity] # this should live here in case we overwrite self_value
		self.stored_likelihood = None
		POSTERIOR_CALL_COUNTER = 0
	
	def set_value(self, v): 
		""" Sets the (self.)value of this hypothesis to v"""
		self.value = v
		
	def copy(self):
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
class FunctionHypothesis(Hypothesis):
	"""
		A special type of hypothesis whose value is a function. 
		The function is automatically eval-ed when we set_value, and is automatically hidden and unhidden when we pickle
		This can also be called like a function, as in fh(data)!
	"""
	
	def __init__(self, v=None, f=None, args=['x']):
		"""
			v - the value of this hypothesis
			f - defaultly None, in which case this uses self.value2function
			args - the argumetns to the function
		"""
		self.args = args # must come first since below calls value2function
		Hypothesis.__init__(self,v) # this initializes prior and likleihood variables, so keep it here!
		self.set_value(v,f)
		
	def copy(self):
		""" Create a copy, only deeply of of v """
		return FunctionHypothesis(v=self.value.copy(), f=self.fvalue, args=self.args)
		
	def __call__(self, *vals):
		""" 
			Make this callable just like a function. Yay python! 
		"""
		return self.fvalue(*vals)
			
	
	def value2function(self, v):
		#print "ARGS=", self.args
		""" How we convert a value into a function. Default is LOTlib.Miscellaneous.evaluate_expression """
		return evaluate_expression(v, args=self.args)
	
	def reset_function(self):
		""" re-construct the function from the value -- useful after pickling """
		self.set_value(self.value)
		
	
	def set_value(self, v, f=None):
		"""
		The the value. You optionally can send f, and not write (this is for some speed considerations) but you better be sure f is correct
		since an error will not be caught!
		"""
		
		Hypothesis.set_value(self,v)
		if f is not None: self.fvalue = f
		elif v is None:   self.fvalue = None
		else:             self.fvalue = self.value2function(v)
	
	# ~~~~~~~~~
	# Evaluate this function on some data
	def get_function_responses(self, data):
		""" 
		Returns a list of my responses to data, handling exceptions (setting to None)
		"""
		out = []
		for di in data:
			#print ">>", di, di.__class__.__name__, type(di), isinstance(di, FunctionData)
			r = None
			try:
				if isinstance(di, FunctionData):    r = self(*di.args)
				elif isinstance(di, UtteranceData): r = self(*di.context)
				else:                               r = self(*di) # otherwise just pass along
			except RecursionDepthException: pass # If there is a recursion depth exception, just ignore (so r=None)
			
			out.append(r) # so we get "None" when things mess up
		return out
		
	# ~~~~~~~~~
	# Make this thing pickleable
	def __getstate__(self):
		""" We copy the current dict so that when we pickle, we destroy the function"""
		dd = copy(self.__dict__)
		dd['fvalue'] = None # clear the function out
		return dd
	def __setstate__(self, state):
		self.__dict__.update(state)
		self.set_value(self.value) # just re-set the value so that we re-compute the function

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LOTHypothesis(FunctionHypothesis):
	"""
		A FunctionHypothesis built from a grammar.
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions.
		
	"""
	
	def __init__(self, G, v=None, f=None, start='START', ALPHA=0.9, rrPrior=False, rrAlpha=1.0, maxnodes=25, ll_decay=0.0, prior_temperature=1.0, args=['x']):
		"""
			G - a grammar
			start - how the grammar starts to generate
			rrPrior - whether we use RR prior or log probability
			f - if specified, we don't recompile the whole function
		"""
		
		# save all of our keywords (though we don't need v)
		self.__dict__.update(locals())
		if v is None: v = G.generate(self.start)
		
		FunctionHypothesis.__init__(self, v=v, f=f, args=args)
		
		self.likelihood = 0.0
		
	def copy(self):
		"""
			Return a copy -- must copy all the other values too (alpha, rrPrior, etc) 
		"""
		assert isinstance(self.value, FunctionNode)
		return LOTHypothesis(self.G, v=self.value.copy(), start=self.start, ALPHA=self.ALPHA, rrPrior=self.rrPrior, rrAlpha=self.rrAlpha, maxnodes=self.maxnodes, args=self.args, ll_decay=self.ll_decay)
		
			
	def propose(self): 
		p = self.copy()
		ph, fb = self.G.propose(self.value)
		p.set_value(ph)
		return p, fb
		
	def compute_prior(self): 
		"""
		
		"""
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			# compute the prior with either RR or not.
			if self.rrPrior: self.prior = self.G.RR_prior(self.value, alpha=self.rrAlpha) / self.prior_temperature
			else:            self.prior = self.value.log_probability() / self.prior_temperature
			
			self.lp = self.prior + self.likelihood
			
		return self.prior
		
	def compute_likelihood(self, data):
		"""
			Computes the likelihood of data.
			The data here is from LOTlib.Data and is of the type FunctionData
		"""
		
		# set up to store this data
		self.stored_likelihood = [None] * len(data)
		
		# compute responses to all data
		responses = self.get_function_responses(data) # get my response to each object
		
		N = len(data)
		self.likelihood = 0.0
		for i in xrange(N):
			r = responses[i]
			di = data[i]
			
			# the pointsiwe undecayed likelihood for this data point
			self.stored_likelihood[i] = log( self.ALPHA*(r==di.output) + (1.0-self.ALPHA)/2.0 )
			
			# the total culmulative decayed likeliood
			self.likelihood += self.stored_likelihood[i] * self.likelihood_decay_function(i, N, self.ll_decay)
		
		self.lp = self.prior + self.likelihood
		
		return self.likelihood
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		for k in G.enumerate_pointwise(self.value):
			yield LOTHypothesis(v=k)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GaussianLOTHypothesis(LOTHypothesis):
	"""
		Like LOTHypothesis but has a Gaussian likelihood
	"""
	
	def __init__(self, G, ll_sd=1.0, prior_temperature=1.0, ll_decay=0.0): 
		""" kwargs should include ll_sd """
		LOTHypothesis.__init__(self, G)
		self.__dict__.update(locals())
		
	def copy(self):
		""" Return a copy -- must copy all the other values too (alpha, rrPrior, etc) """
		assert isinstance(self.value, FunctionNode)
		return GaussianLOTHypothesis(G=self.G, ll_sd=self.ll_sd, prior_temperature=self.prior_temperature, ll_decay=self.ll_decay)
				
	
	def compute_likelihood(self, data):
		""" Compute the likelihood with a Gaussian"""
		
		# set up to store this data
		self.stored_likelihood = [None] * len(data)
		
		# compute responses to all data
		responses = self.get_function_responses(data) # get my response to each object
		
		N = len(data)
		self.likelihood = 0.0
		for i in xrange(N):
			self.stored_likelihood[i] = normlogpdf(responses[i], data[i].output, self.ll_sd)
			
			# the total culmulative decayed likeliood
			self.likelihood += self.stored_likelihood[i] * self.likelihood_decay_function(i, N, self.ll_decay)
		
		self.lp = self.prior + self.likelihood
		
		return self.likelihood

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SimpleGenerativeHypothesis(LOTHypothesis):
	"""
		Here, the data is a hash from strings to counts. 
		Each function eval results in a string, and the likelihood is the probability of generating the 
		observed data. The function eval results in a thunk
		
		NOTE: FOR NOW, Insert/Delete moves are taken off because of some weirdness with the lambda thunks
	"""
	def __init__(self, G): 
		""" kwargs should include ll_sd """
		LOTHypothesis.__init__(self, G, args=[])
		self.__dict__.update(locals())
		
	def copy(self):
		""" Return a copy -- must copy all the other values too (alpha, rrPrior, etc) """
		assert isinstance(self.value, FunctionNode)
		return SimpleGenerativeHypothesis(G=self.G)
	
	def compute_likelihood(self, data, nsamples=500, sm=1e-3):
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
		
		


	