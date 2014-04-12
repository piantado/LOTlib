	
"""
	An attempt at a better search algorithm: 

"""
import LOTlib
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import *
from copy import copy

from LOTlib import lot_iter

def search(h, data, tree_parts, depth, top=100):
	"""
		tree_parts: a dictionary from nonterminal types to a list of potential tree parts           
	"""
	if depth <= 0: return
	
	# Take a node
	beam = FiniteBestSet(N=top)
	
	for ni, n in enumerate(lot_iter(h.value)):
		oldn = n.__copy__(shallow=True) # for restoring at the end?
		
		
		n.setto(copy(n)) # make a new modifiable copy
		for t in lot_iter(tree_parts[n.returntype]):
			n.setto(copy(t)) 
			
			h.reset_function() # recompile the function since we've edited it!
			posterior = sum(h.compute_posterior(data))
			#print "\t", posterior, copy(h)
			
			beam.add(copy(h), p=posterior)
		
		n.setto(oldn) # restore this
		
		#print "BEAM:"
		#for k in lot_iter(beam.get_all(sorted=True)):
			#print sum(k.compute_posterior(data)), k.posterior_score, h, "\t", k
		
		
	for y in lot_iter(beam.get_all(sorted=True)):
		
		yield y
		
		for yprime in lot_iter(search(y, data, tree_parts, depth-1, top=top)):
			yield yprime
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
	
	
	from LOTlib.Examples.Number.Shared import *
	
	DATA_SIZE = 500
	
	data = generate_data(DATA_SIZE)

	# make some trees:
	tree_parts = defaultdict(set)
	for k in xrange(100):
		t = G.generate('WORD')
		for ti in t:
			tree_parts[ti.returntype].add(ti)
			
	# A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
	h0 = NumberExpression(G)

	for i, x in lot_iter(enumerate(search(h0, data, tree_parts, 3))):
		print i, sum(x.compute_posterior(data)), x.posterior_score, get_knower_pattern(x), x
		pass
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	