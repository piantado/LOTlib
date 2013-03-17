

from LOTlib.bvPCFG import *

G = PCFG()

"""
	I,S,K basis
"""
#G.add_rule('START', 'apply', ['EXPR', 'x'], 1.0) # everything is some expression applied to x

G.add_rule('EXPR', 'I', [], 1.0)
G.add_rule('EXPR', 'S', [], 1.0)
G.add_rule('EXPR', 'K', [], 1.0)

G.add_rule('EXPR', 'apply', ['EXPR', 'EXPR'], 2.0)

for x in G.increment_tree('EXPR', 10):
	print x
