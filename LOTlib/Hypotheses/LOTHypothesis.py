"""
	A FunctionHypothesis built from a grammar.		
"""



from FunctionHypothesis import FunctionHypothesis
from copy import copy, deepcopy
from LOTlib.Proposals.RegenerationProposal import RegenerationProposal
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import FunctionData

class LOTHypothesis(FunctionHypothesis):
	"""
		A FunctionHypothesis built from a grammar.		
	"""
	
	def __init__(self, grammar, value=None, f=None, start='START', ALPHA=0.9, maxnodes=25, args=['x'], proposal_function=None, **kwargs):
		"""
			*grammar* - The grammar for the hypothesis (specified in Grammar.py)

			*value* - the value for the hypothesis

			*f* - if specified, we don't recompile the whole function

			*start* - The start symbol for the grammar

			*ALPHA* - parameter for compute_single_likelihood that

			*maxnodes* - the maximum amount of nodes that the grammar can have

			*args* - The arguments to the function

			*proposal_function* - function that tells the program how to transition from one tree to another
			(by default, it uses the RegenerationProposal function)
		"""
		
		# save all of our keywords (though we don't need v)
		self.__dict__.update(locals())
		if value is None: value = grammar.generate(self.start)
		
		FunctionHypothesis.__init__(self, value=value, f=f, args=args, **kwargs)
		# Save a proposal function
		## TODO: How to handle this in copying?
		if proposal_function is None:
			self.proposal_function = RegenerationProposal(self.grammar)
		
		self.likelihood = 0.0
	
	def type(self): return self.value.type()
	
	def set_proposal_function(self, f):
		"""
			Just a setter to create the proposal function
		"""
		self.proposal_function = f
	
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
		"""
			Computes a very similar derivation from the current derivation, using the proposal function we specified
			as an option when we created an instance of LOTHypothesis
		"""
		ret = self.proposal_function(self, **kwargs)
		ret[0].posterior_score = "<must compute posterior!>" # Catch use of proposal.posterior_score, without posteriors!
		return ret
	
	def compute_prior(self): 
		"""
			Compute the log of the prior probability
		"""
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			# compute the prior with either RR or not.
			self.prior = self.value.log_probability() / self.prior_temperature
			
		self.posterior_score = self.prior + self.likelihood
			
		return self.prior
		
	#def compute_likelihood(self, data): # called in FunctionHypothesis.compute_likelihood
	def compute_single_likelihood(self, datum):
		"""
			Computes the likelihood of the data

			The data here is from LOTlib.Data and is of the type FunctionData
			This assumes binary function data -- maybe it should be a BernoulliLOTHypothesis
		"""
		assert isinstance(datum, FunctionData)
		
		return log( self.ALPHA*(self(*datum.input)==datum.output) + (1.0-self.ALPHA)/2.0 )
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		"""
			Returns a generator, where the elements in the generator are instances of LOTHypothesis
			 that can be gotten to from the current LOTHypothesis.
		"""
		for k in grammar.enumerate_pointwise(self.value):
			yield LOTHypothesis(self.grammar, value=k)
