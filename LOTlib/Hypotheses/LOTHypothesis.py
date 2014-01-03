from FunctionHypothesis import FunctionHypothesis
from copy import copy, deepcopy
from LOTlib.Proposals import RegenerationProposal
from LOTlib.Miscellaneous import *

class LOTHypothesis(FunctionHypothesis):
	"""
		A FunctionHypothesis built from a grammar.
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions.
		
	"""
	
	def __init__(self, grammar, v=None, f=None, start='START', ALPHA=0.9, rrPrior=False, rrAlpha=1.0, maxnodes=25, ll_decay=0.0, prior_temperature=1.0, args=['x']):
		"""
			grammar - a grammar
			start - how the grammar starts to generate
			rrPrior - whether we use RR prior or log probability
			f - if specified, we don't recompile the whole function
		"""
		
		# save all of our keywords (though we don't need v)
		self.__dict__.update(locals())
		if v is None: v = grammar.generate(self.start)
		
		FunctionHypothesis.__init__(self, v=v, f=f, args=args)
		
		# Save a proposal function
		## TODO: How to handle this in copying?
		self.proposal_function = RegenerationProposal(self.grammar)
		
		self.likelihood = 0.0
		
	def __copy__(self):
		"""
			Return a copy of myself.
			This makes a deepcopy of everything except grammar (which is the, presumably, static grammar)
		"""

		# Since this is inherited, call the constructor on everything but grammar
		thecopy = type(self)(self.grammar) # call my own constructor with grammar
		
		thecopy.proposal_function = self.proposal_function
		
		# And then deepcopy evrything but grammar
		for k in self.__dict__.keys():
			if k is 'grammar': pass
			if k is 'proposal_function': pass
			else:
				thecopy.__dict__[k] = deepcopy(self.__dict__[k])
			
		return thecopy

			
	def propose(self): 
		p,fb = self.proposal_function(self)
		p.lp = "<must compute posterior!>" # Catch use of proposal.lp, without posteriors!
		return [p,fb]
	
	def compute_prior(self): 
		"""
		
		"""
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			# compute the prior with either RR or not.
			if self.rrPrior: self.prior = self.grammar.RR_prior(self.value, alpha=self.rrAlpha) / self.prior_temperature
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
		for k in grammar.enumerate_pointwise(self.value):
			yield LOTHypothesis(self.grammar, v=k)
