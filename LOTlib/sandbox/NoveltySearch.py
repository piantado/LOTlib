import heapq
from collections import defaultdict
import random

import LOTlib
from LOTlib.FiniteBestSet import FiniteBestSet

def novelty_search(h0s, data, grammar, props=10, novelty_advantage=100):
	"""
		Search through hypotheses, maintaining a queue of good ones. We propose 
		to ones based on their posterior and how much they've been proposed to in the past. 
		See heapweight(h) below -- it determines how we trade off posterior and prior search there...
		
		SO: You are searched further if you are good, and in a "novel" part of the space
		
		TODO: We could make this track the performance of proposals from a given hypothesis?
	"""
	
	novelty = defaultdict(float) # last time we proposed here, what proportion were novel? If we haven't done any, set to 1.0
	froms = defaultdict(int) # how many times did we propose from this?
	tos   = defaultdict(int) # how many to this?
	FS = FiniteBestSet(N=10)
	
	# When we add something to the heap, what weight does it have?
	# This should prefer high log probability, but also it should 
	# keep us from looking at things too much
	def heapweight(h):
		return -h.lp - novelty[h]*novelty_advantage

	openset = []
	for h0 in h0s:
		if h0 not in novelty:
			h0.compute_posterior(data)
			heapq.heappush( openset, (heapweight(h0),h0) )
			novelty[h0] = 1.0 # treat as totally novel
			FS.add(h0, h0.lp)
	
	
	while not LOTlib.SIG_INTERRUPTED:
		lph, h = heapq.heappop(openset)
		
		froms[h] += 1
		
		#proposals_from[h] += props
		print "\n"
		print len(openset), "\t", h.lp, "\t", heapweight(h), "\t", novelty[h], "\t", froms[h], tos[h], "\t", q(h)
		for x in FS.get_all(sorted=True):
			print "\t", x.lp, "\t", heapweight(x), "\t", novelty[x], "\t", froms[x], tos[x],"\t", q(x)
		
		# Store all together so we know who to update (we make their novelty the same as their parent's)
		proposals = [ h.propose()[0] for i in xrange(props) ]
		new_proposals = [] # which are new?
		
		novelprop = 0
		for p in proposals:
			if p not in novelty:
				p.compute_posterior(data)
				FS.add(p, p.lp)
				novelty[p] = "ERROR" # just keep track -- should be overwritten later
				novelprop += 1
				new_proposals.append(p)
			tos[p] += 1
		
		novelty[h] = float(novelprop) / float(props)
		
		# use the novelty from the parent
		for p in new_proposals: 
			novelty[p] = random() * novelty[h]
			heapq.heappush(openset, (heapweight(p), p) )
		
		# and put myself back on the heap, but with the new proposal numbers
		heapq.heappush(openset, (heapweight(h), h) )
		


if __name__ == "__main__":
	
	
	from LOTlib.Examples.Number.Shared import *
	
	DATA_SIZE = 500
	N_INITIAL = 1000
	
	data = generate_data(DATA_SIZE)
	
	h0s = set([ NumberExpression(G) for x in xrange(N_INITIAL) ])
	
	i = 0
	novelty_search(h0s, data, G)
	