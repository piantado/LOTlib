"""
	Just create some combinators and reduce them
"""

from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

from LOTlib.Miscellaneous import q
from LOTlib.Miscellaneous import cons_ # for evaling

G = Grammar()

G.add_rule('START', 'cons_', ['START', 'START'], 2.0)

G.add_rule('START', 'I', None, 1.0)
G.add_rule('START', 'S', None, 1.0)
G.add_rule('START', 'K', None, 1.0)

from LOTlib.CombinatoryLogic import combinator_reduce, CombinatorReduceException

for _ in range(10000):
	t = G.generate()
	
	lst = t.liststring()
	
	print lst, "\t->\t",
	try:
		print combinator_reduce(lst)
	except CombinatorReduceException as e:
		print "NON-HALT"
	