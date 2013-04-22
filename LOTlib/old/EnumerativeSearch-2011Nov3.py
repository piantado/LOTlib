# -*- coding: utf-8 -*-
import heapq

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
	"""
	
	def __init__(self, N=-1, max=False): 
		self.Q = []
		self.N = N
		if max: self.max=-1 # so we can just multiply by this
		else:   self.max=1
		
	def push(self, x, v):
		
		heapq.heappush(self.Q, QueueItem(x, self.max * v))
		
		while self.N > 0 and len(self.Q) > self.N: # and enfornce the size bound; TODO: CHECK THIS
			del self.Q[len(self.Q)-1]
	
	def peek(self):
		return self.Q[0]
	
	def pop(self):
		return heapq.heappop(self.Q).value
		
	def pop_both(self):
		"""
			Returns both value and priority
		"""
		x = heapq.heappop(self.Q)
		return x.value, x.priority * self.max # invert self.max
	
	def __len__(self):
		return len(self.Q)
	
	
	
def enumerative_search( start, next_states, score, N=-1, max=False):
	"""
		state is the current state
		
		next_states is a function mapping states to a list of potential next states
		
		score - maps states to values, lower is better. If +inf, the state is illegal
	"""
	
	# an function to score and potentially push a value
	def score_and_push(x):
		rk = repr(x)
		if rk not in V:
			p = score(x)
			if p < float("+inf"):
				Q.push(x, p)
				V.add(repr(x))
			
	## Main code:
	Q = PriorityQueue(N=N, max=max) # the priority queue
	V = set() # a set of states we've seen already
	
	# add the start state
	score_and_push(start)
	
	## Main loop:
	while len(Q) > 0: # while there is something in the queue
		
		s,p = Q.pop_both()
		
		yield s,p
		
		for k in next_states(s): score_and_push(k)
		


### Testing:
#from copy import copy

#def possible_next_states(s):
	#""" 
		#S is an array and we'll enumerate the changes coordinatewise
	#"""
	#for l in xrange(len(s)):
		#for v in xrange(10):
			#x = copy(s)
			#x[l] = v
			#yield x

#def myscore(x):
	#target = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
	#s = 0.0
	#for i in xrange(len(target)):
		#s += abs(x[i] - target[i])
	#return s
	
#for v in enumerative_search( [0] * 15, possible_next_states, myscore, N=2):
	#print v
