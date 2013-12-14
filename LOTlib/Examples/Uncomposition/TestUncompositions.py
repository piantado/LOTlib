
"""
	Test out how we can un-compose, via a grammar
"""

from LOTlib.Grammar import Grammar
from LOTlib.Hypothesis import *

G = Grammar()
G.BV_WEIGHT = 2.0 # probability weight for our introduced variables

G.add_rule('BOOL', 'x', None, 10.0)

G.add_rule('BOOL', 'and', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'or', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'not', ['BOOL'], 1.0)

G.add_rule('BOOL', 'exists', ['FUNCTION', 'SET'], 0.50)
G.add_rule('BOOL', 'forall', ['FUNCTION', 'SET'], 0.50) 

G.add_rule('SET', 'S', None, 1.0)

G.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv=["BOOL"]) # bvtype means we introduce a bound variable below

G.display_rules()

for i in xrange(100):
	t = G.generate('BOOL')
	print t
		
	for a,b in G.all_simple_uncompositions(t):
		print a, "\t", b
	print "\n"
		
	