# -*- coding: utf-8 -*-
"""

	Implement a RR-style boolean expression. 
	
"""
from LOTlib.Hypothesis import *
from LOTlib.bvPCFG import *
from LOTlib.BasicPrimitives import *

class BooleanExpression(FunctionHypothesis):
	"""
		Implement a Rational Rules (Goodman et al 2008)-style grammar over Boolean expressions
	"""
	
	def __init__(self, G, v=None, start='BOOL', ALPHA=0.9, rrPrior=False, maxnodes=25, ll_decay=0.0):
		"""
			G - a grammar
			start - how the grammar starts to generate
			rrPrior - whether we use RR prior or log probability
		
		"""
		Hypothesis.__init__(self)
		
		# save all of our keywords (though we don't need v)
		self.__dict__.update(locals())
		
		if v is None: self.set_value(G.generate(self.start))
		else:         self.set_value( v )
		
		self.likelihood = 0.0
		
	def copy(self):
		"""
			Return a copy -- must copy all the other values too (alpha, rrPrior, etc) 
		"""
		return BooleanExpression(self.G, v=self.value.copy(), start=self.start, ALPHA=self.ALPHA, rrPrior=self.rrPrior, maxnodes=self.maxnodes)
			
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
			if self.rrPrior: self.prior = self.G.RR_prior(self.value)
			else:            self.prior = self.value.log_probability()
			
			self.lp = self.prior + self.likelihood
			
		return self.prior
		
	def compute_likelihood(self, data, decay=0.0):
		"""
			Computes the likelihood of data.
			Here the data is a list of [ T/F, object ], [T/F, object], etc.
		"""
		
		# set up to store this data
		self.stored_likelihood = [None] * len(data)
		
		# compute responses to all data
		responses = self.get_function_responses(map(lambda x: x[1], data)) # get my response to each object
		
		N = len(data)
		self.likelihood = 0.0
		for i in xrange(N):
			r = responses[i]
			di = data[i]
			
			# the pointsiwe undecayed likelihood for this data point
			self.stored_likelihood[i] = log( self.ALPHA*(r==di[0]) + (1.0-self.ALPHA)/2.0 )
			
			# the total culmulative decayed likeliood
			self.likelihood += self.stored_likelihood[i] * self.likelihood_decay_function(i, N, decay)
		
		self.lp = self.prior + self.likelihood
		
		return self.likelihood
	