"""
	--- There's one problem which is that sometimes with L, we end up with a function that recursively returns a
	-- We should do an explore/exploit tradeoff explicitly -- with some probabiliyt, you sample one with similar semantics, 
	   and with some other probability, you sample very different
	   
	   --> Hmm maybe we need to make proposals specifically to explain one of the data points
	   
	   --> NOTE: We have not checked at all detailed balance here --this was all unused, quick code to see what's up. 
	   
	   
	   --> I think what's going on here is that our proposals don't take into account the priors, and so you end  up not being able to overcome the priors with fb
	   
"""

from LOTlib.Examples.Number.Shared import *
from collections import defaultdict
import numpy
import random
from math import log
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram, dendrogram

from matplotlib.pyplot import show # For plotting the dendrogram

def set_parents(x):
	""" Takes a ClusterNode and add a "parent" field"""
	l = x.get_left()
	if l is not None: 
		setattr(l, "parent", x)
		set_parents(l)
	
	r = x.get_right()
	if r is not None: 
		setattr(r, "parent", x)
		set_parents(r)

subtrees = defaultdict(set)
for i in xrange(1000):
	t = NumberExpression(G)
	for x in t.value:
		if x.count_subnodes() < 10 and not x.contains_function("L_"): # don't add recursion, that's a mess
			#print "ADDING ", x.returntype, x, t
			subtrees[ x.returntype ].add(x)
		
#quit()
test_data = generate_data(300)


cluster_root = dict() # hash each nonterminal type to the root of the clusters
tree2ClusterNode = dict() # hash each tree to the cluster node containing it
nt2N = dict()
id2set = defaultdict(dict) 
for k in subtrees.keys():
	
	the_trees = [x for x in subtrees[k]] # get a fixed ordering here
	nt2N[k] = len(the_trees )
	 
	leaves = defaultdict(set) # each leaf is a set of nodes with items identical on the data
	
	resp2set = dict()
	for t in the_trees:
		resp = tuple(map(str, NumberExpression(G, v=t).get_function_responses(test_data))) # make all responses "strings" for easier comparison
		#print k, t
		if resp not in resp2set: resp2set[resp] = [t]
		else:                    resp2set[resp].append(t)
		
	
	# compute the distance matrix, now between elements of resp2set
	distance = numpy.zeros( (len(resp2set), len(resp2set)), dtype=float )
	responses = resp2set.keys()
	for i1 in xrange(len(responses)):
		r1 = responses[i1]
		for i2 in xrange(i1):
			r2 = responses[i2]
			distance[i1][i2] = numpy.sum( numpy.array([ r1[i] == r2[i] for i in xrange(len(r1)) ], dtype=bool ))
			distance[i2][i1] = distance[i1][i2]
	
	# form a linkage matrix, and convert it to a scipy.cluster.hierarchy.ClusterNode
	cluster_root[k] = to_tree(linkage(distance)) 
	
	# Go through and make "parent" nodes so we can go up in sampling
	set_parents(cluster_root[k])
	setattr(cluster_root[k], "parent", cluster_root[k])
	
	# Map each input tree thing to the appropriate ClusterNode
	def tree2ClusterNode_mapper(x):
		if x.is_leaf(): 
			id2set[k][x.id] = resp2set[responses[x.id]] # save this set so we can use it later
			for t in id2set[k][x.id]:
				tree2ClusterNode[ t ] = x
	cluster_root[k].pre_order( tree2ClusterNode_mapper )

print "# Built proximity dendrogram!"

def proximity_proposal(h, G, p=0.5, p_resample=0.5): # propose to a nodes

	# Flip a coin for which proposal to use?
	# We must mix with normal RR proposals since this is by itself not ergodic
	if random.random() < p_resample:
		return h.propose()
	
	t = h.value
	
	# Else, do this:
	Z = G.resample_normalizer(t) # the total probability
	
	# copy since we modify in place
	newt = copy(t)
	ni = G.sample_random_node(newt)
	
	# Now we do our actual resampling
	if ni in tree2ClusterNode:
		root = cluster_root[ni.returntype]
		
		a = tree2ClusterNode[ni] # "a" is where we started
		r = numpy.random.geometric(p)-1 # how far up we go?
		
		top = a # "top" is how far we went up and then down
		for i in xrange( r ): top = top.parent # move up
		
		b = top # "b" is what we end up sampling
		br = 0 # what random number would we have to choose from b?
		while not b.is_leaf(): 
			b = (b.get_left() if (random.random() < 0.5) else b.get_right())
			br += 1
		
		#print "R=", r,  id2set[ni.returntype][a.id] is id2set[ni.returntype][b.id]
		
		asize = len(id2set[ni.returntype][a.id])
		bsize = len(id2set[ni.returntype][b.id])
		
		btree = sample1( id2set[ni.returntype][b.id] )
		
		#print id2set[ni.returntype][b.id]
		#print ni
		#print btree
		#print "\n\n"
		
		# Count up the probability of going up higher than top, and then coming back down to top 
		# and then to a,b
		x = top # Stand at the top, and compute the prob of going a->b, and also b->a
		nup = 0 # How far up have we gone?
		forward, backward = -inf, -inf
		while True:
			# prob of going up to top and above, and then back down to b
			forward  = logplusexp(forward, geometric_ldensity(r+nup+1,  p) + br*log(0.5) + -log(bsize))
			
			# up to top and above, and then down to a
			backward = logplusexp(backward, geometric_ldensity(br+nup+1, p) + r*log(0.5) + -log(asize))
			
			if x is root: break # exit before incrementing below
			else:         x = x.parent
			x = x.parent 
			nup += 1 # how far up are we from top?
			
		# Actually do the replacement
		ni.setto( copy(btree))
		
		newZ = G.resample_normalizer(newt)
		#print "FB = ", forward, (log(ni.resample_p) - log(Z)), backward, (log(ni.resample_p) - log(newZ)) 
		f = (log(ni.resample_p) - log(Z))    + forward
		b = (log(ni.resample_p) - log(newZ)) + backward
		
		#print f,b, asize, bsize, newt
		#print t
		#print newt
		#print "\n\n"
		
	else: 
		return proximity_proposal(h, G, p, p_resample) # A bad idea? Or should we just do below and pass?
		#f,b = 0.0, 0.0
		#pass # do nothing if we're not in the cache!
				
	return NumberExpression(G,v=newt), f-b

	
data = generate_data(50)
initial_hyp = NumberExpression(G)

target = dict()
for h in LOTlib.MetropolisHastings.mh_sample( copy(initial_hyp), data, 150000):
	target[copy(h)] = h.lp
#for h in pickle.load(open("/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/runs/run-201Oct14.pkl")).get_all(): # Load a target set
	#h.compute_posterior(data)
	#target[h] = h.lp

	
print "# Done generating target hypotheses"

from LOTlib.Evaluation import evaluate_sampler
test_sampler =  LOTlib.MetropolisHastings.mh_sample( copy(initial_hyp), data) # LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, 100000, proposer=lambda x: proximity_proposal(x,G), skip=5, trace=False)
evaluate_sampler(target, test_sampler, trace=False)



#data = generate_data(300)
#data = test_data
#initial_hyp = NumberExpression(G)
#for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, 100000, proposer=lambda x: proximity_proposal(x,G), skip=5, trace=False):
	#print h.lp, q(get_knower_pattern(h)), h.compute_prior(), h.compute_likelihood(data), q(h)
	#pass

