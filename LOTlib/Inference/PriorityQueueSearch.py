"""

    Search inspired by monte carlo tree search


The "priority" of a tree will be its log probability MINUS a penalty for how many samples have been drawn

"""
from LOTlib.PriorityQueue import PriorityQueue, QueueItem

from LOTlib.Examples.Number.Model.Inference import make_h0, generate_data

initial_N = 100 # how many to generate?
nsamples  = 100

data = generate_data(100)

# Create an initial set of trees
initial_set = set()
while len(initial_set) < initial_N:
    initial_set.add(make_h0())

# update all posteriors
for h in initial_set:
    h.compute_posterior(data)

# Now keep them in a priority queue based on how much we think we can get from them
import heapq
Q = PriorityQueue([QueueItem(h, h.posterior_score) for h in initial_set])
while len(Q) > 0:
    t = Q.pop()

    proposals = [ t.propose for _ in xrange(nsamples)]

    # Set the priority and add proposals depending on

