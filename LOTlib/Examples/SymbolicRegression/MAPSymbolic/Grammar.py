"""
    Define a grammar here. Same as SymbolicRegression grammar, but with added constants
"""

from LOTlib.Examples.SymbolicRegression.Grammar import grammar

NCONSTANTS = 4
CONSTANT_NAMES = ['C%i'%i for i in xrange(NCONSTANTS) ]

# Supplement the grammar with constant names
for c in CONSTANT_NAMES:
    grammar.add_rule('EXPR', c, None, 5.0)