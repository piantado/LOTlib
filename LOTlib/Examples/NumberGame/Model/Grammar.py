from LOTlib.Grammar import Grammar

# Default parameters for integer primitives
TERMINAL_PRIOR = .25
INTEGERS       = [1,   2,  3,  4,  5,  6,  7,  8,  9, 10]
INTEGERS_DIST  = [1., 1., 1., .8, 1., .8, 1., .6, 1., .8]   # len same as INT_PRIMITIVES


# Setting up our LOT hypothesis grammar
grammar = Grammar()

# Sets
grammar.add_rule('START', '', ['SET'], 1)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1)

# Range of numbers, e.g. [1,100] (numbers 1 through 100)
grammar.add_rule('SET', 'range_', ['EXPR', 'EXPR'], 10)
# Mapping expressions over sets of numbers
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1, bv_type='EXPR')
grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 1)

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], .1)

# Terminals
for i in range(0, len(INTEGERS)):
    grammar.add_rule('EXPR', str(INTEGERS[i]), None, TERMINAL_PRIOR * INTEGERS_DIST[i])

