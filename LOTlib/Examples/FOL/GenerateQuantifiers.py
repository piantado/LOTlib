
"""
	An example of generating quantified logic with lambdas. See FOL.py for inference about first-order logic
"""

from LOTlib.Grammar import Grammar

# Create a  grammar:
grammar = Grammar()

grammar.add_rule('BOOL', 'x', None, 2.0) # X is a terminal, so arguments=None

 # Each of these is a function, requiring some arguments of some type
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'exists_', ['FUNCTION', 'SET'], 0.50)
grammar.add_rule('BOOL', 'forall_', ['FUNCTION', 'SET'], 0.50) 

grammar.add_rule('SET', 'S', None, 1.0)

# And allow us to create a new kind of function
grammar.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type='BOOL', bv_args=None) # bvtype means we introduce a bound variable below
grammar.BV_WEIGHT = 2.0 # When we introduce bound variables, they have this (relative) probability


for i in xrange(1000):
	x = grammar.generate('BOOL')
	
	print x.log_probability(), x
