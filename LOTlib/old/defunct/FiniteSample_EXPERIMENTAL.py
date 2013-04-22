# -*- coding: utf-8 -*-
"""
	This takes a sampler and creates a finite set of samples, keeping track of the top N (where N can be Infinity)
	
	This is implemented using a heap in order to efficiently store the top N samples we've seen so far
	
	This stores self.parmeters, which keesp track of what parameters were run when this this sampled. 
	
	NOTE: The heap here stores the hypotheses, and the dict stores str(h), so that we only have one copy of hyps
"""
from scipy.maxentropy import logsumexp
from copy import deepcopy

from LOTlib.Miscellaneous import *
from LOTlib.PriorityQueue import *



"""

	Maybe what should happen is that PriorityQueue takes an argument name specifying how to refer to the priority. 
	And its default can use QueueItem, but it can also use "lp" or whatever else we want
	
	Maybe we can also have a way to increment a count on each element too. So each element would get added and sorted by an attribute we define, and get a "count" increment we define too for each time it is added
	
	No, we want a hash table that checks access and therefore also 
	
	pq = PriorityQueue()
	pq.add( [1,2,3] )


	PriorityQueueSet -- stores a set of the top things in a queue, 






"""


class PriorityQueueCompressedSet:
	
	"""
		Stores a priority queue of elements (keeping the top N) and a dictionary to store counts (how often you have been added to the set). 
		
		You can pop, but you pop one a ta time from the set
		You also can zero the counts (to get just a set)
		You can iterate and pop tokens
		Zero the counts 
		You can merge ("populate") from another one or from a set or from a priorityQueue
		
		
		You can recompute the priorities -- do this by passing a function (and add this functionality to proirity queue)
		- Add functionality to take the top N
		e.g.: PQ.recompute_priorities(lambda x: x.compute_posterior(data))
		
		
		HMM WE NEED TO CHANGE PRIORITY QUEUE SO THAT WHEN IT OVERFLOWS, YOU GET THE THING BACK HERE TO REMOVE FROM D
	"""
	
	def __init__(self, N=-1):
		
		self.D = dict() 
		self.Q = PriorityQueue()
		pass
	
	def add(self, x, v):
		
		# the count of how many are here
		#NOTE: We do NOT check the priority of this that we stored earlier
		if x in D: 
			self.D[x] += 1
		else:      
			self.D[x]  = 1
			self.Q.push(x,v) # add to the queue
	
	def peek(self):
		return self.Q[0].value
		
	def pop(self):
		top = self.Q.peek()
		if self.D[
		heapq.heappop(self.Q).value







class FiniteSample(PriorityQueue):
	"""
		This is a special kind of priority queue that stores the top hypotheses, and allows efficient recomputation
	"""
	
	def __init__(self, N=-1):
		pass
	
	# for each hypothesis, recompute its posterior
	def compute_posterior( self, data, reheapify=True):
		for x in self.Q: 
			x.priority = x.value.compute_posterior(data)
		if reheapify: self.reheapify() # recompute the heap since the lps are updated, Not needed if we just use the whole sample
		
	def show(self):
		keys = sorted(self.Q, key=lambda x: x.priority)
		for k in keys: print  k.priority, "\t", k.value
		
	# this can take a finite same, a dictionary, or a generator and populate the hypotheses
	# by adding counts
	def populate(self, frm):
		if isinstance(frm, FiniteSample):
			for k in frm.Q: self.add_sample(k.value) 
		else:
			for s in frm: self.add_sample(s)
				
	# this is what technically adds the sample, dealing with the heap, self.count, etc. 
	def add_sample(self, s, cnt=1):
		
		# do not add these
		#if math.isnan(s.lp) or math.isinf(s.lp): return
		
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
	def hypotheses(self): return self.Q # don't return the dict because that may not have been updated
	
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