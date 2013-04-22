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
		A class for representing learning single words - a SimpleLexicon is one where the primary inferences are
		over individual word meanings
	
		You may overwrite the weightfunction, which maps *[h f] to a positive number corresponding to the probability of producing a word h, f
		It defaults to lambda x: 1.0
	
		TODO: we can probably make this faster by not passing around the context sets so much!
		
	"""
	
	def __init__(self, G, args, alpha=0.90, palpha=0.90):
		Hypothesis.__init__(self)
		self.dexpr = dict()
		self.dfunc = dict()
		self.grammar = G
		self.args = args
		self.alpha = alpha
		self.palpha = palpha
		
	def copy(self):
		"""
			A *much* faster copy
			TODO: We don't have to recompile dfunc -- we can just point to the existing one -- much faster
		"""
		new = SimpleLexicon(self.grammar, self.args, alpha=self.alpha, palpha=self.palpha)
		for w in self.dexpr.keys():
			new.set_word(w, self.dexpr[w].copy(), f=self.dfunc[w]) # copy this guy
		return new
		
	def __str__(self):
		out = ''
		for w, v in self.dexpr.iteritems():
			out = out + str(w) + ':\t' + str(v) + '\n'
		return out
		
	def __eq__(self, other):
		return (str(self) == str(other))
	def __hash__(self): return hash(str(self))
	def __cmp__(self, other): return cmp(str(self), str(other))
		
	## For pickling: remove the "function" or else pickle gets angry
	def __getstate__(self):
		self.clear_functions()
		return self.__dict__
	def __setstate(self, state):
		self.__dict__.update(state)
		self.unclear_functions()
	
	# this sets the word and automatically compute its function
	def set_word(self, w, v, f=None):
		"""
			This sets word w to value v.
			You can optionally supply f if you don't want it to be recomputed (this is faster)
			but it had better be right!
		"""
		self.dexpr[w] = v
		if f == None: self.dfunc[w] = evaluate_expression(v, self.args)
		else:         self.dfunc[w] = f

	# set this function (to make something to sample from)
	def force_function(self, w, f):
		self.dexpr[w] = None
		self.dfunc[w] = f
		
	def all_words(self):
		return self.dexpr.keys()
	
	def clear_functions(self):
		"""
			Call this before being pickled to clear functions (which cannot be pickled)
		"""
		self.dfunc = dict() # clear these
	
	def unclear_functions(self):
		"""
			Call this after being pickled to restore all of the functions (which, btw, cannot be pickled)
		"""
		for k in self.dexpr.keys():
			self.dfunc[k] = evaluate_expression(self.dexpr[k], self.args)
	
	def weightfunction(self, h, f):
		return 1.0
		
	###################################################################################
	## MH stuff
	###################################################################################
	
	def propose(self):
		new = self.copy()
		w = weighted_sample(self.dexpr.keys()) # the word to change
		p,fb = self.grammar.propose( self.dexpr[w] )
		
		new.set_word(w, p)
		
		return new, fb

	def compute_prior(self):
		self.prior = sum([v.log_probability() for v in self.dexpr.values()])
		self.lp = self.prior + self.likelihood
		return self.prior		
	
	## Here is an old form that is pretty slow because self.weightfunction_utt is called multiple times when this is 
	# evaluated on many data points. The below likelihood is much faster
	# score a single utterance from a set of utterances
	# here, the data is a list of [utterance, possible_utterances]
	#def compute_likelihood(self, data):
		#self.likelihood = 0.0
		#for u, pu in data:
			#self.likelihood = self.likelihood + self.score_utterance(u, pu)
		#self.lp = self.prior + self.likelihood
		#return self.likelihood
		
	# This combines score_utterance with likelihood so that everything is much faster
	def compute_likelihood(self, data):
		self.likelihood = 0.0
		
		weights = dict()
		for w in self.dexpr.keys():
			weights[w] = self.weightfunction(self.dexpr[w], self.dfunc[w])
			
		# inline computation of likelihood -- much faster (?)
		self.likelihood = 0.0
		for u, pu in data:
			
			f = self.dfunc[u.word](*u.context)
			t,m = self.partition_utterances(pu)
			
			## This is the slow part! You should score a bunch of utterances at once!
			all_weights = sum(map( lambda u: weights[u.word], pu))
			true_weights = sum(map( lambda u: weights[u.word], t))
			met_weights = sum(map( lambda u: weights[u.word], m))
			
			p = 0.0
			w = weights[u.word] # the weight of the one we have
			if f == True:    p = self.palpha * self.alpha * w / true_weights + self.palpha * (1.0 - self.alpha) * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
			elif f == False: p = ifelse(true_weights==0, 1.0, 1.0-self.alpha) * self.palpha * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
			else:            p = ifelse(met_weights==0, 1.0, (1.0 - self.palpha)) * w / all_weights
			
			self.likelihood += log(p)
			
		self.lp = self.prior + self.likelihood
		return self.likelihood
			
		
	###################################################################################
	## Sampling and dealing with responses
	###################################################################################
	
	# a wrapper so we can call this easily on utterances
	def weightfunction_utt(self, u):
		return self.weightfunction(self.dexpr[u.word], self.dfunc[u.word])
		
	def weightfunction_word(self, w):
		return self.weightfunction(self.dexpr[w], self.dfunc[w])
		
	
	# Take a set of utterances and partition them into true/met
	def partition_utterances(self, us):
		"""
			Partition utterances, returning those that are true, and those with the presup met	
			NOTE: The commended code is much slower
		"""
			
		trues, mets = [], []
		for u in us:
			#print u.word, self.dexpr[u.word]
			fa = self.dfunc[u.word](*u.context) # apply f to the arguments we got
			if fa is True:    
				trues.append(u)
				mets.append(u)
			elif fa is False: 
				mets.append(u)
			else: pass # not met
			
		return [trues, mets]
	
		## OLD VERSION:
		# apply to everything
#		app = [ self.dfunc[u.word](*u.context) for u in us ]
#		# and now partition the utterances. This is much slower than the 
#		# below commented out code!
#		trues = [ us[i] for i in xrange(len(app)) if app[i] is True ]
#		mets  = [ us[i] for i in xrange(len(app)) if (app[i] is True or app[i] is False) ]
#		return [trues, mets]
	
	# take a set of utterances and sample them according to our probability model
	def sample_utterance(self, us):
		
		t,m = self.partition_utterances(us)
		
		if flip(self.palpha) and (len(m) > 0): # if we sample from a presup is true
			if (flip(self.alpha) and (len(t)>0)):
				#print map(lambda x: x.word, t)
				#print map(self.weightfunction_utt, t)
				#print map(self.weightfunction_utt, t) / numpy.sum(map(self.weightfunction_utt, t))
				return weighted_sample(t, probs=map(self.weightfunction_utt, t), log=False)
			else:   return weighted_sample(m, probs=map(self.weightfunction_utt, m), log=False)
		else: return weighted_sample(us, probs=map(self.weightfunction_utt, us), log=False) # sample from all utterances
			
	# This scores a single utterance, but does it slowly so use the above likelihood function (Which only computes the weights once)
	def score_utterance(self, u, us):
		
		finp = self.dfunc[u.word](*u.context)
		t,m = self.partition_utterances(us)
		
		## This is the slow part! You should score a bunch of utterances at once!
		all_weights = sum(map( lambda u: self.weightfunction_utt(u), us))
		true_weights = sum(map( lambda u: self.weightfunction_utt(u), t))
		met_weights = sum(map( lambda u: self.weightfunction_utt(u), m))
		
		##OLD VERSION -- PRE 2012 Mar 10
		#p = 0.0
		#w = self.weightfunction_utt(u) # the weight of the one we have
		#if finp == True:    p = self.palpha * self.alpha * w / true_weights + self.palpha * (1.0 - self.alpha) * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
		#elif finp == False: p = ifelse(true_weights==0, 1.0, self.alpha) * self.palpha * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
		#else:               p = ifelse(met_weights==0, 1.0, (1.0 - self.palpha)) * w / all_weights
		
		p = 0.0
		w = self.weightfunction_utt(u) # the weight of the one we have
		
		if finp == True:    p = self.palpha * self.alpha * w / true_weights + self.palpha * (1.0 - self.alpha) * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
		elif finp == False: p = ifelse(true_weights==0, 1.0, 1.0-self.alpha) * self.palpha * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
		else:               p = ifelse(met_weights==0, 1.0, (1.0 - self.palpha)) * w / all_weights
		
		return log(p)
		
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		
		for w in self.all_words():
			for k in self.grammar.enumerate_pointwise(self.dexpr[w]):
				new = self.copy()
				new.set_word(w, k)
				yield new
	
	

class VectorizedLexicon(Hypothesis):
	"""
		This is a Lexicon class that stores *only* indices into my_finite_trees, and is designed for doing gibbs,
		sampling over each possible word meaning. It requires running EnumerateTrees.py to create a finite set of 
		trees, and then loading them here. Then, all inferences etc. are vectorized for super speed.
		
		This requires a bunch of variables (the global ones) from run, so it cannot portably be extracted. But it can't
		be defined inside run or else it won't pickle correctly. Ah, python. 
		
		NOTE: This takes a weirdo form for the likelihood function
	"""
	
	def __init__(self, target_words, finite_trees, priorlist, word_idx=None, ALPHA=0.75, PALPHA=0.75):
		Hypothesis.__init__(self)
		
		self.__dict__.update(locals())
		
		self.N = len(target_words)
		
		if self.word_idx is None: 
			self.word_idx = np.array( [randint(0,len(self.priorlist)-1) for i in xrange(self.N) ])
		
	def __repr__(self): return str(self)
	def __eq__(self, other): return np.all(self.word_idx == other.word_idx)
	def __hash__(self): return hash(tuple(self.word_idx))
	def __cmp__(self, other): return cmp(str(self), str(other))
	
	def __str__(self): 
		s = ''
		aw = self.target_words
		for i in xrange(len(self.word_idx)):
			s = s + aw[i] + "\t" + str(self.finite_trees[self.word_idx[i]]) + "\n"
		s = s + '\n'
		return s
	
	def copy(self):
		return VectorizedLexicon(self.target_words, self.finite_trees, self.priorlist, word_idx=np.copy(self.word_idx), ALPHA=self.ALPHA, PALPHA=self.PALPHA)
	
	def enumerative_proposer(self, wd):
		for k in xrange(len(self.finite_trees)):
			new = self.copy()
			new.word_idx[wd] = k
			yield new
			
	def compute_prior(self):
		self.prior = sum([self.priorlist[x] for x in self.word_idx])
		self.lp = self.prior + self.likelihood
		return self.prior
	
	def compute_likelihood(self, data):
		"""
			Compute the likelihood on the data, super fast
			
			These use the global variables defined by run below.
			The alternative is to define this class inside of run, but that causes pickling errors
		"""
		
		response,weights,uttered_word_index = data # unpack the data
		
		# vector 0,1,2, ... number of columsn
		zerothroughcols = np.array(range(len(uttered_word_index)))
		
		r = response[self.word_idx] # gives vectors of responses to each data point
		w = weights[self.word_idx]
		
		if r.shape[1] == 0:  # if no data
			self.likelihood = 0.0
		else:
			
			true_weights = ((r > 0).transpose() * w).transpose().sum(axis=0)
			met_weights  = ((np.abs(r) > 0).transpose() * w).transpose().sum(axis=0)
			all_weights = w.sum(axis=0)
			rutt = r[uttered_word_index, zerothroughcols] # return value of the word actually uttered
			
			## now compute the likelihood:
			lp = np.sum( np.log( self.PALPHA*self.ALPHA*weights[self.word_idx[uttered_word_index]]*(rutt>0) / (1e-5 + true_weights) + \
					self.PALPHA*( (true_weights>0)*(1.0-self.ALPHA) + (true_weights==0)*1.0) * weights[self.word_idx[uttered_word_index]] * (np.abs(rutt) > 0) / (1e-5 + met_weights) + \
					( (met_weights>0)*(1.0-self.PALPHA) + (met_weights==0)*1.0 ) * weights[self.word_idx[uttered_word_index]] / (1e-5 + all_weights)))
				
			self.likelihood = lp
		self.lp = self.likelihood+self.prior
		return self.likelihood 
		
	def propose(self):
		print "*** Cannot propose to VectorizedLexicon"
		assert False