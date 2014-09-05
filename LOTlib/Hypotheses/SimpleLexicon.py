# -*- coding: utf-8 -*-

"""

	TODO:
		- Make the.valueicon be indexable like an array/dict, rather than having to say h.value[...] say h[..]
		

"""
from copy import copy
from inspect import isroutine

from LOTlib.Miscellaneous import *
from LOTlib.Hypotheses.Hypothesis import Hypothesis

class SimpleLexicon(Hypothesis):
	"""
		A class for mapping words to hypotheses. 
		
		By itself this has no likelihood function. 
		
		TODO: we can probably make this faster by not passing around the context sets so much.
		
	"""
	def __init__(self, make_hypothesis, words=(), **kwargs):
		"""
			hypothesis - a function to generate hypotheses
			words -- words to initially add (sampling from the prior)
		"""
		Hypothesis.__init__(self, value=dict(), **kwargs)
		self.__dict__.update(locals())
		
		assert isroutine(make_hypothesis) # check that we can call
		
		# update with the supplied words, each generating from the grammar
		for w in words:
			self.set_word(w, v=None)
			
	def __copy__(self):
		""" Copy a.valueicon. We don't re-create the fucntions since that's unnecessary and slow"""
		new = type(self)(self.make_hypothesis, words=copy(self.words))
		for w in self.value.keys():
			new.value[w] = copy(self.value[w])
			
		# And copy everything else
		for k in self.__dict__.keys():
			if k not in ['self', 'value']:
				new.__dict__[k] = copy(self.__dict__[k])
				
		return new
		
	def __str__(self):
		"""
			This defaultly puts a \0 at the end so that we can sort -z if we want (e.g. if we print out a posterior first) 
		"""
		return '\n'.join([ str(w)+':\t'+str(v) for w,v in sorted(self.value.iteritems())]) + '\0'
	def __hash__(self): return hash(str(self))
	def __eq__(self, other):   return (str(self)==str(other)) # simple but there are probably better ways
	
	# this sets the word and automatically compute its function
	def set_word(self, w, v=None):
		""" 
			This sets word w to value v. v can be either None, a FunctionNode or a  Hypothesis, and
			in either case it is copied here. When it is a Hypothesis, the value is extracted. If it is 
			None, we generate from the grammar
		"""
		
		# Conver to standard expressiosn
		if v is None: 
			v = self.make_hypothesis()
		
		assert isinstance(v, Hypothesis)
			
		self.value[w] = v
		
	def all_words(self): 
		return self.value.keys()
	
	
	def force_function(self, w, f):
		"""
			Allow force_function
		"""
		self.value[w].force_function(f)
		
	###################################################################################
	## MH stuff
	###################################################################################
	
	def propose(self):
		new = copy(self)
		
		w = weighted_sample(self.value.keys()) # the word to change
		p,fb = self.value[w].propose()
		
		new.set_word(w, p)
		
		return new, fb

		
	def compute_prior(self):
		self.prior = sum([x.compute_prior() for x in self.value.values()]) / self.prior_temperature
		self.posterior_score = self.prior + self.likelihood
		return self.prior		
			
	# This combines score_utterance with likelihood so that everything is much faster
	def compute_single_likelihood(self, di):
		raise NotImplementedError

