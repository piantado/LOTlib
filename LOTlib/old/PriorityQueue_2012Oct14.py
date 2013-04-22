# -*- coding: utf-8 -*-

"""
		TODO: 2011 Nov 17 -- Delete is not working from unique priority queue --- sometimes it is not in the set???

"""
import heapq
import operator

from LOTlib.Miscellaneous import *
from collections import deque
from copy import deepcopy

class QueueItem:
	"""
		A wrapper to hold items and scores in the queue--just wraps "cmp" on a priority value
	"""
	def __init__(self, v, p):
		self.value = v
		self.priority = p
	
	def __cmp__(self, y):
		return cmp(self.priority, y.priority)
	
	def __repr__(self): return repr(self.value)
	def __str__(self):  return str(self.value)



class PriorityQueue:
	"""
		A priority queue, potentially of bounded size N (when N > 0).
		The default is a min heap, but passing in max=True makes a max heap.
		
		TODO: Add a "attribute" flag that lets you use a priority queue based on an attribute rather than another object
	"""
	
	def __init__(self, N=-1, max=False): 
		self.Q = []
		self.N = N
		if max: self.max=-1 # so we can just multiply by this
		else:   self.max=1
		self.worst = float("inf")
	
	def __contains__(self, x):
		return (x in self.Q)
		
	def __iter__(self):
		for x in self.get_all(): yield x
	
	def reheapify(self): 
		"""
			If you change the priorities, call this to make it back into a PriorityQueue
		"""
		heapq.heapify(self.Q)
	
	def delete_worst(self):
		"""
			Deletes the worst element and returns it
		"""
		worst_index, worst_value = max(enumerate([a.priority for a in self.Q]), key=operator.itemgetter(1))
		self.worst = worst_value # could do a little better by finding the end again, but lets not do that since its slower
		
		# potentially slow for lots of bound violations -- TODO: Replace this with faster heap operations
		w = self.Q[worst_index] 
		del self.Q[worst_index] 
		heapq.heapify(self.Q)
		return w
		
	def push(self, x, v):
		"""
			TODO: Fix this so that we know whether the thing we added was just deleted or not. Otherwise, this doesn't quite make sense...
		"""
		
		#print "Pushing ", x, v, self.worst, self.max
		l = len(self.Q)
		
		# push on if we are lower than the worst
		if (l == 0) or (self.N < 0) or (l < self.N) or (self.max * v < self.worst):
			heapq.heappush(self.Q, QueueItem(x, self.max * v))
			# and enforce the size bound
			if (self.N > 0) and (l+1 > self.N):
				w = self.delete_worst()
			return True
		return False
	
	def get_all(self):
		return [ a.value for a in self.Q ]
	
	def get_sorted(self, decreasing=False):
		c = deepcopy(self.Q)
		c.sort( reverse = not decreasing)
		return [ a.value for a in c ]
	
	def peek(self):
		return self.Q[0].value, self.Q[0].priority * self.max
	
	def pop(self): return self.pop_both()[0]
	def pop_both(self):
		"""
			Returns both value and priority
		"""
		x = heapq.heappop(self.Q)
		return x.value, x.priority * self.max # invert self.max
		
	
	def __len__(self):
		return len(self.Q)
	
	
	def sample(self, log=True):
		""" 
			Sample from this priority queue.
			if log=True, then our priorities are treated as log probabilities
		"""
		s = weighted_sample(self.Q, probs=map( lambda x: x.priority * self.max, self.Q), log=log)
		return s.value
	
	def merge(self, y):
		"""
			Copy over everything from y.
			This is slightly inefficient becuase we create all new QueueItems, but it's easiest to deal with min/max
		"""
		for yi in y.Q:
			self.push(yi.value, yi.priority * y.max)
		
	def save(self, f):
		out_file = open(f, 'wb')
		pickle.dump(self, out_file)
		out_file.close()
		
	def show(self):
		"""
			TODO: Define a good show method here..
		"""
		pass
	
###########################################

class UniquePriorityQueue(PriorityQueue):
	"""
		Here is a priority queue that stores a hash for uniqueness so that we only store the top N unique objects
		in priority order. 
		
		NOTE: This *assumes* that we will NEVER get a duplicate with a *different* priority
	"""
	def __init__(self, N=-1, max=False):
		self.s = set() # for storing uniqueness - a set of items to be added
		PriorityQueue.__init__(self, N, max=max)
	
	def push(self, x, v):
		
		for k in self.s: print k is None,

		# if we are not in the set and we are added (return value from PriorityQueue.push)
		if (x not in self.s):
			if PriorityQueue.push(self, x, v):
				self.s.add(x)
				return True
		return False
			
	def delete_worst(self):
		w = PriorityQueue.delete_worst(self)
		#print ">>", w
		## TODO: Okay there can be an error when we add and then immediately remove an element, so here we have to check if w.value in self.s, but we shouldn't have to
		if w.value in self.s:
			#print "==>", w, type(w), w.value in self.s, self.s
			self.s.remove(w)
	
	def pop_both(self):
		x,v = PriorityQueue.pop_both(self)
		self.s.remove(x)
		return x,v
		

#PQ = PriorityQueue(N=4, max=False)

#for a in xrange(100):
	#PQ.push(a, random.random())
	#print [(a.value, a.priority) for a in PQ.Q ]