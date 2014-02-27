from FunctionHypothesis import FunctionHypothesis
from copy import copy, deepcopy
from LOTlib.Proposals import RegenerationProposal
from LOTlib.Miscellaneous import *

class LOTHypothesis(FunctionHypothesis):
	"""
		A FunctionHypothesis built from a grammar.
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions.
		
	"""
	
	def __init__(self, grammar, value=None, f=None, start='START', ALPHA=0.9, rrPrior=False, rrAlpha=1.0, maxnodes=25, ll_decay=0.0, prior_temperature=1.0, likelihood_temperature=1.0, args=['x'], proposal_function=None):
		"""
			grammar - a grammar
			start - how the grammar starts to generate
			rrPrior - whether we use RR prior or log probability
			f - if specified, we don't recompile the whole function
		"""
		
		# save all of our keywords (though we don't need v)
		self.__dict__.update(locals())
		if value is None: value = grammar.generate(self.start)
		
		FunctionHypothesis.__init__(self, value=value, f=f, args=args)
		# Save a proposal function
		## TODO: How to handle this in copying?
		if proposal_function is None:
			self.proposal_function = RegenerationProposal(self.grammar)
		
		self.likelihood = 0.0
	
	def __copy__(self):
		"""
			Return a copy of myself.
			This makes a deepcopy of everything except grammar (which is the, presumably, static grammar)
		"""

		# Since this is inherited, call the constructor on everything, copying what should be copied
		thecopy = type(self)(self.grammar, value=copy(self.value), f=self.f, proposal_function=self.proposal_function) 
		
		# And then then copy the rest
		for k in self.__dict__.keys():
			if k not in ['self', 'grammar', 'value', 'proposal_function', 'f']:
				thecopy.__dict__[k] = copy(self.__dict__[k])
		
		return thecopy

			
	def propose(self, **kwargs): 
		ret = self.proposal_function(self, **kwargs)
		ret[0].posterior_score = "<must compute posterior!>" # Catch use of proposal.posterior_score, without posteriors!
		return ret
	
	def compute_prior(self): 
		"""
		
		"""
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			# compute the prior with either RR or not.
			if self.rrPrior: self.prior = self.grammar.RR_prior(self.value, alpha=self.rrAlpha) / self.prior_temperature
			else:            self.prior = self.value.log_probability() / self.prior_temperature
			
			self.posterior_score = self.prior + self.likelihood
			
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
			self.stored_likelihood[i] = log( self.ALPHA*(r==di.output) + (1.0-self.ALPHA)/2.0 ) / self.likelihood_temperature
			
			# the total culmulative decayed likeliood
			self.likelihood += self.stored_likelihood[i] * self.likelihood_decay_function(i, N, self.ll_decay)
				
		
		self.posterior_score = self.prior + self.likelihood
		
		return self.likelihood
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		for k in grammar.enumerate_pointwise(self.value):
			yield LOTHypothesis(self.grammar, value=k)
