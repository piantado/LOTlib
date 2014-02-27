# -*- coding: utf-8 -*-
"""
	Shared functions and variables for the number learning model. 
"""

import LOTlib
from LOTlib.Grammar import Grammar
from LOTlib.BasicPrimitives import *
import LOTlib.Inference.ParallelTempering
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from random import randint

ALPHA = 0.75 # the probability of uttering something true
GAMMA = -30.0 # the log probability penalty for recursion
LG_1MGAMMA = log(1.0-exp(GAMMA)) # TODO: Make numerically better
USE_RR_PRIOR = False # Use the Rational Rules prior? If false, we just use log probability under the PCFG. NOTE: Using it is not supported under pypy

WORDS = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']

########################################################################################################
## Define a PCFG

## The priors here are somewhat hierarchical by type in generation, tuned to be a little more efficient
## (but the actual RR prior does not care about these probabilities)

G = Grammar()

G.add_rule('BOOL', 'and_',    ['BOOL', 'BOOL'], 1./3.)
G.add_rule('BOOL', 'or_',     ['BOOL', 'BOOL'], 1./3.)
G.add_rule('BOOL', 'not_',    ['BOOL'], 1./3.)

G.add_rule('BOOL', 'True',    None, 1.0/2.)
G.add_rule('BOOL', 'False',   None, 1.0/2.)

## note that this can take basically any types for return values
G.add_rule('WORD', 'if_',    ['BOOL', 'WORD', 'WORD'], 0.5)
G.add_rule('WORD', 'ifU_',    ['BOOL', 'WORD'], 0.5) # if returning undef if condition not met

G.add_rule('BOOL', 'cardinality1_',    ['SET'], 1.0)
G.add_rule('BOOL', 'cardinality2_',    ['SET'], 1.0)
G.add_rule('BOOL', 'cardinality3_',    ['SET'], 1.0)

G.add_rule('BOOL', 'equal_',    ['WORD', 'WORD'], 1.0)

G.add_rule('SET', 'union_',     ['SET', 'SET'], 1./3.)
G.add_rule('SET', 'intersection_',     ['SET', 'SET'], 1./3.)
G.add_rule('SET', 'setdifference_',     ['SET', 'SET'], 1./3.)
G.add_rule('SET', 'select_',     ['SET'], 1.0)

G.add_rule('SET', 'x',     None, 4.0)

G.add_rule('WORD', 'L_',        ['SET'], 1.0) 

G.add_rule('WORD', 'next_', ['WORD'], 1.0)
G.add_rule('WORD', 'prev_', ['WORD'], 1.0)

#G.add_rule('WORD', 'undef', [], 1.0)
G.add_rule('WORD', 'one_', None, 0.10)
G.add_rule('WORD', 'two_', None, 0.10)
G.add_rule('WORD', 'three_', None, 0.10)
G.add_rule('WORD', 'four_', None, 0.10)
G.add_rule('WORD', 'five_', None, 0.10)
G.add_rule('WORD', 'six_', None, 0.10)
G.add_rule('WORD', 'seven_', None, 0.10)
G.add_rule('WORD', 'eight_', None, 0.10)
G.add_rule('WORD', 'nine_', None, 0.10)
G.add_rule('WORD', 'ten_', None, 0.10)

##########################################################
#Define a class for running MH

class NumberExpression(LOTHypothesis):
	#__module__ = os.path.splitext(os.path.basename(__file__))[0]  # So that when we pickle this, we know where to read from
 	
	def __init__(self, G, value=None, f=None, proposal_function=None, **kwargs): 
		LOTHypothesis.__init__(self,G,proposal_function=proposal_function, **kwargs)
		
		if value is None: self.set_value(G.generate('WORD'), f)
		else:             self.set_value(value, f)
		
	def copy(self):
		""" Must define this else we return "FunctionHypothesis" as a copy. We need to return a NumberExpression """
		return NumberExpression(G, value=self.value.copy(), prior_temperature=self.prior_temperature)
		
	def compute_prior(self): 
		"""
			Compute the number model prior
		"""
		if self.value.count_nodes() > 20:
			self.prior = -Infinity
		else: 
			if self.value.contains_function("L_"): recursion_penalty = GAMMA
			else:                                  recursion_penalty = LG_1MGAMMA
			
			if USE_RR_PRIOR: # compute the prior with either RR or not.
				self.prior = (recursion_penalty + G.RR_prior(self.value))  / self.prior_temperature
			else:
				self.prior = (recursion_penalty + self.value.log_probability())  / self.prior_temperature
			
			self.posterior_score = self.prior + self.likelihood
			
		return self.prior
	
	def compute_single_likelihood(self, datum, response):
		"""
			Computes the likelihood of data.
			TODO: Make sure this precisely matches the number paper. 
			
		"""
		if response == 'undef' or response == None: 
			return log(1.0/10.0) # if undefined, just sample from a base distribution
		else:   return log( (1.0 - ALPHA)/10.0 + ALPHA * ( response == datum.output ) )
		
	
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		for k in G.enumerate_pointwise(self.value):
			yield NumberExpression(value=k)
	

# # # # # # # # # # # # # # # # # # # # # # # # #
# The target

#target = NumberExpression("one_ if cardinality1_(x) else next_(L_(setdifference_(x, select_(x))))") # we need to translate "if" ourselves
#target = NumberExpression(value="if_(cardinality1_(x), one_, two_)")

# NOTE: Not necessary, but only for testing -- these are discovered in the real model via search
#one_knower   = NumberExpression("one_ if cardinality1_(x) else undef") 
#two_knower   = NumberExpression("one_ if cardinality1_(x) else ( two_ if cardinality2_(x) else undef )") 
#three_knower = NumberExpression("one_ if cardinality1_(x) else ( two_ if (cardinality2_(x) ) else ( three_ if (cardinality3_(x) else undef) )") 
#four_knower  = NumberExpression("one_ if cardinality1_(x) else ( two_ if (cardinality2_(x) ) else ( three_ if (cardinality3_(x) else (four_ if (cardinality4_(x) else undef) ) )") 

def get_knower_pattern(ne):
	"""
		This computes a string describing the behavior of this knower-level
	"""
	out = ''
	mydata = [ FunctionData( [set(sample_sets_of_objects(n, all_objects))], '') for n in xrange(1,10) ] 
	resp = ne.get_function_responses( mydata )
	return ''.join([ str(word_to_number[x]) if (x is not None and x is not 'undef' ) else 'U' for x in resp])
	

def generate_data(data_size):
	"""
		Sample some data according to the target
	"""
	data = []
	for i in range(data_size):
		# how many in this set
		set_size = weighted_sample( range(1,10+1), probs=[7187, 1484, 593, 334, 297, 165, 151, 86, 105, 112] )
		# get the objects in the current set
		s = set(sample_sets_of_objects(set_size, all_objects))
		
		# sample according to the target
		if random() < ALPHA: r = WORDS[len(s)-1]
		else:                r = weighted_sample( WORDS )
		
		# and append the sampled utterance
		data.append(FunctionData( input=[s], output=r) ) # convert to "FunctionData" and store
	return data

	
# # # # # # # # # # # # # # # # # # # # # # # # #
# All objects -- not very exciting

#here this is really just a dummy -- one type of object, which is replicated in sample_sets_of_objects
all_objects = make_all_objects(shape=['duck'])  

# all possible data sets on 10 objects
all_possible_data = [ ('', set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10) ] 

