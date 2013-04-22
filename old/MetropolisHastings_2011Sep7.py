# -*- coding: utf-8 -*-

"""
	proposer - should return a *new copy* of a hypothesis, and a forward-backward probability
"""
from random import random
from math import log
from LL_Misc import *

def display_samplefunction(x):
	print x
	
"""
	This wraps a hypothesis with a prior and likelihood, but for hashing, strings, it looks just like the hypothesis
	 This is slightly fancier in that you can set the "lp" 
	 Do NOT mutate the prior and likelihood; we can change lp as much as we want for sampling
"""
class Sample:
	def __init__(self, h, prior, likelihood):
		self.prior = prior
		self.likelihood = likelihood
		self.h = h
		self.lp = prior + likelihood # for sampling. Note: this can be set!
	
	def __str__(self):
		#return ''.join([str(self.likelihood), "\t", str(self.prior), "\t",  str(self.h)])
		return str(self.h)
		
	# for hashing Samples, just hash hypotheses
	def __hash__(self): return hash(self.h)
	def __cmp__(self, x): return cmp(str(self.h), str(x))

"""
	A stub class for a sampler. Iniitalize with the functions for computing prior and likelihood, and 
"""
class Sampler:
	
	def __init__(self, prior=None, likelihood=None):
		self.prior = prior
		self.likelihood = likelihood
		self.current_sample = Sample(None, -Infinity, -Infinity)
		
	def sample(self,  h, data, steps, skip=0, trace=False):
		pass
	
	
	
"""
	This takes a sampler and creates a finite set of samples. 
	These can be renormalized if we want, or 
"""
from scipy.maxentropy import logsumexp
class FiniteSample:
	
	def __init__(self):
		self.d = dict()
		self.count = 0 # how many total samples
	
	def show(self):
		for k,v in self.d.iteritems():
			print v, "\t", k
	
	def populate(self, sampler, h, data, steps, skip=0):
	
		self.count = self.count + steps
		
		for s in sampler.sample(h, data, steps, skip, False):
			#print s, hash(s), (s in self.d)
			#if not s in self.d: self.d[s] = 1
			#else: self.d[s] = self.d[s] + 1
			hashplus(self.d, s) # add one to this thing's count
	
	# here s is a list of samples we glom on
	def push_samples(self, s):
		for si in s:
			hashplus(self.d, si)
		
	# compute the normalizer via logsumexp
	def normalizer(self):
		return logsumexp( [ x.lp for x in self.d.iterkeys()] )
		
	# this actually mutates the hypothese to normalize them
	# this would be bad to change keys, but since the lp has nothing to do with
	# sample keys, we are okay
	def normalize(self):
		Z = self.normalizer()
		for k in self.d.iterkeys(): k.lp = k.lp - Z
		return Z
		
	"""
		This bootstraps the proportion of the probability mass we've covered
		It samples from this population and sees how much of it you cover again. 
		NOTE: This only approximates, and is only good if count is large
		
		NOTE: HMM This isn't right, is it?...
	"""
	#def bootstrap_population_coverage(self, bootn):
		
		#bootsample = []
		#Z = self.normalizer()
		
		#hyps = self.d.keys()
		#for b in range(bootn):
			#k = FiniteSample()
			## the hyps have lp, so we can just sample them
			#k.push_samples(weighted_sample(hyps, N=self.count, return_probability=False))
			#bootsample.append( exp(k.normalizer() - Z ) )
			
		#print bootsample
			


"""
	Metropolis-Hastings algorithm
"""

class MetropolisHastingsSampler():
	

class MetropolisHastingsSampler(Sampler):
		
	def __init__(self, prior=None, likelihood=None, proposer=None):
		Sampler.__init__(self, prior, likelihood)
		self.proposer = proposer
		
	# run some number of mcmc steps, generating them
	def sample(self, h, data, steps, skip=0, trace=False):
		
		## NOTE: self.h is used in here since this is saved across runs
		self.current_sample = Sample(h, self.prior(h), self.likelihood(data, h))
		
		for mhi in range(steps):
			for skp in range(skip+1):				
				
				proposal, fb = self.proposer(self.current_sample.h)
				
				new_prior = self.prior(proposal)
				new_likelihood = self.likelihood(data, proposal)
				
				r = (new_prior + new_likelihood) - (self.current_sample.prior + self.current_sample.likelihood) - fb
				
				if trace: print "Proposal: ", proposal, new_prior, new_likelihood, r
				
				if log(random.random()) < r:
					self.current_sample = Sample(proposal, new_prior, new_likelihood)
					
			#if samplefunction is not None: samplefunction(self)
			yield self.current_sample

#class MetropolisHastingsSampler(Sampler):
		
	#def __init__(self, prior=None, likelihood=None, proposer=None):
		#Sampler.__init__(self, prior, likelihood)
		#self.proposer = proposer
		
	## run some number of mcmc steps, generating them
	#def sample(self, h, data, steps, skip=0, trace=False):
		
		### NOTE: self.h is used in here since this is saved across runs
		#self.current_sample = Sample(h, self.prior(h), self.likelihood(data, h))
		
		#for mhi in range(steps):
			#for skp in range(skip+1):				
				
				#proposal, fb = self.proposer(self.current_sample.h)
				
				#new_prior = self.prior(proposal)
				#new_likelihood = self.likelihood(data, proposal)
				
				#r = (new_prior + new_likelihood) - (self.current_sample.prior + self.current_sample.likelihood) - fb
				
				#if trace: print "Proposal: ", proposal, new_prior, new_likelihood, r
				
				#if log(random.random()) < r:
					#self.current_sample = Sample(proposal, new_prior, new_likelihood)
					
			##if samplefunction is not None: samplefunction(self)
			#yield self.current_sample
