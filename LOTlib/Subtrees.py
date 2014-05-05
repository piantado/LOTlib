"""

	Procedures for extracting and manipulating subtrees
	
"""
from LOTlib.Miscellaneous import UniquifyFunction

def generate_trees(grammar, start='START', N=1000):
	"""
		Yield a bunch of unique trees, produced from the grammar
	"""
	for _ in xrange(N):
		yield grammar.generate(start)


@UniquifyFunction
def generate_unique_trees(grammar, start='START', N=1000):
	"""
		Yield a bunch of unique trees, produced from the grammar
	"""
	for _ in xrange(N):
		t = grammar.generate(start)
		yield t

@UniquifyFunction
def generate_unique_complete_subtrees(grammar, start='START', N=1000):
	"""
		genreate from start and yield all seen subtrees
	"""
	for t in generate_unique_trees(grammar, start=start, N=N):
		for ti in t: yield ti
				
@UniquifyFunction	
def generate_unique_partial_subtrees(grammar, start='START', N=1000, npartial=10, p=0.5):
	"""
		Generate from grammar N times, and for each sample npartial partial subtrees with the given p parameter
		from EACH element of t
	"""
	for t in generate_unique_trees(grammar, start=start, N=N):
		
		for ti in t:
			for _ in xrange(npartial):
				yield ti.random_partial_subtree(p=p)
	
# # # # # # # # # # # # # # # # # # # # # # # # #
# Quick helper functions for subtrees

def count_identical_subtrees(t,x):
	"""
		in x, how many are identical to t?
	"""
	return sum([tt==t for tt in x])

def count_identical_nonterminals(t,x):
	""" 
		How many nonterminals in x are of type t?
		
		Here we add up how many nodes have the same return type
		OR how many leaves (from partial trees) have the same returntype
	"""
	
	return sum([tt.returntype==t for tt in x]) +\
	       sum([tt==t for tt in x.all_leaves()])

def count_subtree_matches(t, x):
	return sum(map(lambda tt: tt.partial_subtree_root_match(t), x))
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
	
	
	from LOTlib.Examples.Number.Shared import *
	
	#for t in generate_unique_trees(G, start='WORD'): print t
	
	#for t in generate_unique_complete_subtrees(G, start='WORD'): print t

	for t in generate_unique_partial_subtrees(G, start='WORD'): print t
	