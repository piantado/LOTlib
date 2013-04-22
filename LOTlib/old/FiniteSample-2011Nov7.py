# -*- coding: utf-8 -*-
"""
	This takes a sampler and creates a finite set of samples, keeping track of the top N (where N can be Infinity)
	
	This is implemented using a heap in order to efficiently store the top N samples we've seen so far
	
	This stores self.parmeters, which keesp track of what parameters were run when this this sampled. 
	
	NOTE: The heap here stores the hypotheses, and the dict stores str(h), so that we only have one copy of hyps
"""
from scipy.maxentropy import logsumexp
import heapq # for storing the top N
import pickle
from copy import deepcopy

from LOTlib.Miscellaneous import *

class FiniteSample:
	
	def __init__(self, topN=100, **parameters):
		self.d = dict() # this hashes *string* representations of hypotheses (not the hyps themselves), so there is only one copy of hyps
		self.count = 0 # how many total samples
		self.topN = topN
		self.heap = []
		#self.parameters = dict()
		#self.parameters.update(parameters) # save all the parameters from when we were created
	
	# this takes a list of FiniteSamples and returns a new list, where they have been mereged by 
	# the values of their parameters. 
	# So, you can run a bunch of different jobs and then merge their outputs like this
	#@staticmethod
	#def merge_by_parameters( fsl ):
		#m = dict()
		#for fs in fsl:
			#k = str(fs.parameters) # make a string of the parameters
			#if k in m: m[k].populate(fs)
			#else: m[k] = deepcopy(fs)
		#return m.values() # return each of the keys
			
	def contains(self, k):
		return (k in self.d)
		
	# for each hypothesis, recompute its posterior
	def compute_posterior( self, data, reheapify=True):
		for h in self.heap:
			h.compute_posterior(data)
		if reheapify: self.reheapify() # recompute the heap since the lps are updated, Not needed if we just use the whole sample
		
	def show(self):
		#print map(type, self.heap)
		if len(self.heap) > 0:
			keys = sorted(self.heap, key=lambda x: x.lp)
			for k in keys: print self.d[str(k)], "\t", k.lp, "\t", k
		else: print "*Empty*"
		
	# this can take a finite same, a dictionary, or a generator and populate the hypotheses
	# by adding counts
	def populate(self, frm):
		if isinstance(frm, FiniteSample): 
			for k in frm.heap: self.add_sample(k, cnt=frm.d[str(k)]) 
		#elif isinstance(frm, dict): 
			#for k,v in frm.iteritems(): self.add_sample(k, cnt=v)
		else:
			for s in frm: self.add_sample(s)
				
	# this is what technically adds the sample, dealing with the heap, self.count, etc. 
	def add_sample(self, s, cnt=1):
		"""
			Add a sample to a finite sample. 
			NOTE: If the lp is -inf or nan, this does not add
		"""
		#print "Adding:"
		#print (s in self.d)
		#print s
		#print "==============="
		#for h in self.heap: print h
		#print "\n\n"
		
		# do not add these
		if math.isnan(s.lp) or math.isinf(s.lp): return
		
		sstr = str(s) # stored in the dict
		if sstr in self.d:
			self.d[sstr] += cnt
		else: # not in the dictionary
			if len(self.d) < self.topN:
				heapq.heappush(self.heap, s) # else just push on
				hashplus(self.d, sstr, cnt)
			else:
				bottom = self.heap[0]
				if s.lp <= bottom.lp: pass # do nothin
				else:
					del self.d[str(bottom)]
					heapq.heappush(self.heap, s)
					heapq.heappop(self.heap) # discard the bottom from the heap
					self.d[sstr] = cnt # incorporate this thing
	
	# if you recompute lp on any of the hypotheses, you should call this
	def reheapify(self):
		heapq.heapify(self.heap)
		
	# this is how probability mass is in some subset that we trim
	def trimmed_lp(self, trim):
		return logsumexp( [x.lp for x in heapq.nsmallest(trim, self.heap) ] ) - self.normalizer()
		
	# compute the normalizer via logsumexp
	def normalizer(self):
		return logsumexp( [ x.lp for x in self.heap ] )
	
	# this actually mutates the hypothese to normalize them
	# this would be bad to change keys, but since the lp has nothing to do with
	# sample keys, we are okay
	def normalize(self):
		Z = self.normalizer()
		for x in self.heap: x.lp = x.lp - Z
		return Z
		
	# for saving these hypotheses using pickle
	def save(self, f):
		out_file = open(f, 'wb')
		pickle.dump(self, out_file)
		out_file.close()
	
	# this takes either a directory or a file
	# if it takes a directory, then it loads each file and adds *all* of their samples
	#def load(self, f)
		
	# for everything in our hash and heap, set it's "function" attribute to None
	# this is becuase we can't pickle functions
	#def clear_functions(self):
		#for h in self.heap: h.clear_function()
	#def undo_clear_functions(self):
		#for h in self.heap: h.undo_clear_function()
		
	# get all of the hypotheses here
	def hypotheses(self): return self.heap # don't return the dict because that may not have been updated
	
	# 
	#def set_top_N(self):
		#pass
	#def 
		
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