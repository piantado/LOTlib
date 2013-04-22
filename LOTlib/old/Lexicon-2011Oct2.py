# -*- coding: utf-8 -*-

"""
	This stores a mapping from words to meanings, where each word has the same type
	NOTE: This only allows simple PCFG priors for now
	
	TODO:
		- Add functionality for pickling -- need the ability to zero out functions
"""
from random import sample
from copy import deepcopy

from LOTlib.MetropolisHastings import *
from LOTlib.Hypothesis import *


# a wrapper for utterances -- packages together a word and a context
# our word functions are applied to utterances to yeild a truth value
# (this does not do compositional utterances -- that's why its "Simple")
class SimpleUtterance:
	def __init__(self, w, c):
		self.word = w
		self.context = c
	
	def __repr__(self):
		return str(self.word)+': '+str(self.context)
		
	def __hash__(self):   return hash(str(self))
	def __cmp__(self, x): return cmp(str(self), str(x))

# takes a bunch of contexts and words and crosses to get all utterances
def make_all_SimpleUtterances(words, contexts):
	out = []
	for w in words:
		for c in contexts:
			out.append(SimpleUtterance(w,c))
	return out
	
class SimpleLexicon(Hypothesis):
	"""
	
		TODO: we can probably make this faster by not passing around the context sets so much!
		
	"""
	
	def __init__(self, G, args, alpha=0.7, palpha=0.9):
		Hypothesis.__init__(self)
		self.dexpr = dict()
		self.dfunc = dict()
		self.grammar = G
		self.args = args
		self.alpha = alpha
		self.palpha = palpha
		
	def __str__(self):
		out = ''
		for w, v in self.dexpr.iteritems():
			out = out + str(w) + ':\t' + str(v) + '\n'
		return out
		
	# this sets the word and automatically compute its function
	def set_word(self, w, v):
		self.dexpr[w] = v
		self.dfunc[w] = evaluate_expression(v, self.args)

	# set this function (to make something to sample from)
	def force_function(self, w, f):
		self.dexpr[w] = None
		self.dfunc[w] = f
		
	def all_words(self):
		return self.dexpr.keys()
	
	###################################################################################
	## MH stuff
	###################################################################################
	
	def propose(self):
		new = deepcopy(self)
		w = weighted_sample(self.dexpr.keys()) # the word to change
		p,fb = self.grammar.propose( self.dexpr[w] )
		new.set_word(w, p)
		return new, fb

	def compute_prior(self):
		self.prior = sum([v.log_probability() for v in self.dexpr.values()])
		self.lp = self.prior + self.likelihood
		return self.prior		
		
	# here, the data is a list of [utterance, possible_utterances]
	def compute_likelihood(self, data):
		self.likelihood = 0.0
		for u, pu in data:
			self.likelihood = self.likelihood + self.score_utterance(u, pu)
		self.lp = self.prior + self.likelihood
		return self.likelihood
		
	###################################################################################
	## Sampling and dealing with responses
	###################################################################################
	
	# Take a set of utterances and partition them into true/met
	def partition_utterances(self, us):
		
		trues, mets = [], []
		
		for u in us:
			fa = self.dfunc[u.word](*u.context) # apply f to the arguments we got
			if fa is True:    
				trues.append(u)
				mets.append(u)
			elif fa is False: 
				mets.append(u)
			
		return [trues, mets]
	
	# take a set of utterances and sample them according to our probability model
	def sample_utterance(self, us):
		
		t,m = self.partition_utterances(us)
		
		if flip(self.palpha) and (len(m) > 0): # if we sample from a presup is true
			if (flip(self.alpha) and (len(t)>0)):
				return weighted_sample(t)
			else:   return weighted_sample(m)
			
		else: return weighted_sample(us) # sample from all utterances
		
	# score a single utterance from a set of utterances
	def score_utterance(self, u, us):
		
		finp = self.dfunc[u.word](*u.context)
		t,m = self.partition_utterances(us)
		
		lenall = len(us)
		lentrue = len(t)
		lenmet = len(m)
		
		p = 0.0
		if finp == True:    p = self.palpha * self.alpha / lentrue + self.palpha * (1.0 - self.alpha) / lenmet + (1.0 - self.palpha) / lenall # choose from the trues
		elif finp == False: p = ifelse(lentrue==0, 1.0, self.alpha) * self.palpha / lenmet + (1.0 - self.palpha) / lenall # choose from the trues
		else:               p = ifelse(lenmet==0, 1.0, (1.0 - self.palpha)) / lenall
		
		return log(p)
		
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		for w in self.all_words():
			for k in self.grammar.enumerate_pointwise(self.dexpr[w]):
				new = deepcopy(self)
				new.set_word(w, k)
				yield new
		
		
		
		
		