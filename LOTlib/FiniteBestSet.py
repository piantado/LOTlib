# -*- coding: utf-8 -*-

"""

	This is a version of what was called "PriorityQueue.py" in LOTlib.VERSION < 0.3. 
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
	def __init__(self, x, p):
		self.x = x
		self.priority = p
	
	def __cmp__(self, y):
		# Comparisons are based on priority
		return cmp(self.priority, y.priority)
	
	def __repr__(self): return repr(self.x)
	def __str__(self):  return str(self.x)


class FiniteBestSet:
	"""
		This class stores the top N (possibly infite) hypotheses it observes. It can also make the set of top hypotheses unique
		It works by storing a priority queue (in the opposite order), and popping off the worst as we need to add more
	"""
	
	def __init__(self, N=float("inf"), max=True, unique=True):
		self.__dict__.update(locals())
		
		self.max_multiplier = ifelse(self.max, 1, -1) # invert sign 
		
		self.Q = [] # we use heapq to 
		self.unique_set = set()
		
		
	def __contains__(self, x): 
		return (x in self.Q)
		
	def __iter__(self):
		for x in self.get_all(): yield x	
	
	def __len__(self):
		return len(self.Q)
		
	def push(self, x, p):
		""" Add x with value v to the set """
		
		if self.unique and (x in self.unique_set): 
			return
		else:
			heapq.heappush(self.Q, QueueItem(x, self.max_multiplier*p))
			if self.unique: 	self.unique_set.add(x) # add to the set 
			
			# if we have too many elements
			if len(self) > self.N: 
				rr = heapq.heappop(self.Q)
				if self.unique: self.unique_set.remove(rr.x) # clean out the removed from the set
	
	def get_all(self, **kwargs): 
		""" Return all elements (arbitrary order). Does NOT return a copy. This uses kwargs so that we can call one 'sorted' """
		if kwargs.get('sorted', False):
			return  [ c.x for c in sorted(self.Q, reverse = not kwargs.get('decreasing',False))]
		else:
			return  [ c.x for c in self.Q]
		
	##NOTE: NOW DEFUNCT: USE .get_all
	#def get_sorted(self, decreasing=False): 
		""" Return all elements in sorted order. Returns a *copy*  via 'sorted' """
		#return [ c.x for c in sorted(self.Q, reverse = not decreasing)]
		
	def merge(self, y):
		"""
			Copy over everything from y. Here, y may be a list of things to merge (e.g. other FiniteBestSets)
			This is slightly inefficient becuase we create all new QueueItems, but it's easiest to deal with min/max
		"""
		if isinstance(y, list):
			for yi in y: self.merge(yi)
		else:
			for yi in y.Q:
				self.push(yi.x, yi.priority*y.max_multiplier) # mult y by y.max_multiplier to convert it back to the original scale
	
	def save(self, f):
		# Just a wrapper for pickle that makes saving a little easier
		out_file = open(f, 'wb')
		pickle.dump(self, out_file)
		out_file.close()
		
if __name__ == "__main__":
	
	import random
	
	# Check the max
	for i in xrange(100):
		Q = FiniteBestSet(N=10)

		ar = range(100)
		random.shuffle(ar) 
		for x in ar: Q.push(x,x)
		
		assert set(Q.get_all()).issuperset( set([90,91,92,93,94,95,96,97,98,99]))
		print Q.get_sorted()
	
	# check the min
	for i in xrange(100):
		Q = FiniteBestSet(N=10, max=False)

		ar = range(100)
		random.shuffle(ar) 
		for x in ar: Q.push(x,x)
		
		assert set(Q.get_all()).issuperset( set([0,1,2,3,4,5,6,7,8,9]))
		print Q.get_sorted()
