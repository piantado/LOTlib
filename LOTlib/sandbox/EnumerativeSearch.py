# -*- coding: utf-8 -*-
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import *

## Make this interruptable by CTRL-C
import sys
import signal
SIG_INTERRUPTED = False
def signal_handler(signal, frame): 
	global SIG_INTERRUPTED
	SIG_INTERRUPTED = True
signal.signal(signal.SIGINT, signal_handler)


def enumerative_search( start, next_states, score, N=-1, breakout=float("inf"), yield_immediate=False):
	"""
		start -- either a list or a single element saying what we start with. Generally this works better with a rich starting base
		
		next_states is a function mapping states to a list of potential next states
		
		score - maps states to values, higher is better (UNLIKE priority queue)
		
		breakout -- if the current score is worse than max_so_far - breakout, then don't go further
		            this is useful for doing integration since we can ignore the very tiny. This makes the search considerably more efficient, but may not result in correctness
		
		NOTE / TODO: This has one inefficiency which is that if we find a new path to a state, we just push it on so there are
		             two copies on the stack. The second won't be visited again since it will be put in "visited" the first time it's seen
		             but we could remove it, and then the stack size would be more correct
	"""
	
	## Main code:
	Q = FiniteBestSet(N=N, max=True) # the priority queue
	visited = set() # a set of states we've seen already
	enumerative_search.maxsofar = float("-inf") # ugh a workaround since nested functions can't change this otherwise
	
	# an function to score and potentially push a value
	def score_and_push(k):
		"""
			Score and push, returning the scored value 
		"""
		if k not in visited:
			p = score(k)
			
			if (enumerative_search.maxsofar - breakout > p): return None# if p is really bad, don't append
			
			# Hmm this could go before or after the above breakout line -- for really big spaces, 
			# we should do this after so we don't cache a lot of nonsense; for really slow
			# evaluation functions, this should go before so we don't repeat
			visited.add(k)
			
			if p != None: # if we return None, don't do it
				
				# otherwise push onto queue
				if p > enumerative_search.maxsofar: enumerative_search.maxsofar = p
				Q.push(k, p)
				return p
		
	# add the start state
	if not isinstance(start, list): start = list(start)
	for k in start: 
		p = score_and_push(k)
		if yield_immediate and (p is not None): yield k,p
		
		
	## Main loop:
	while len(Q) > 0 and not SIG_INTERRUPTED: # while there is something in the queue
		
		#print len(Q)
		#print out the current priority queue
		#i = 0
		#for g in Q.Q: 
			#print "   >", g.priority, g.value
			#i += 1
			#if i > 50: break
		#print "\n\n"
		
		x, s = Q.pop_both()
		# if we are yeilding from the top of the queue
		if not yield_immediate: yield x, s
		
		for k in next_states(x): 
			
			p = score_and_push(k)
			
			# if we are yeilding as we score
			if yield_immediate and (p is not None): yield k,p

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

	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	