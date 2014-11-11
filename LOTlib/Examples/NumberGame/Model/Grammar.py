from LOTlib.Grammar import Grammar

# Default parameters for integer primitives
TERMINAL_PRIOR = .25
INTEGERS       = [1,   2,  3,  4,  5,  6,  7,  8,  9, 10]
INTEGERS_DIST  = [1., 1., 1., .8, 1., .8, 1., .6, 1., .8]   # len same as INT_PRIMITIVES


# Setting up our LOT hypothesis grammar
grammar = Grammar()

# Sets
grammar.add_rule('START', '', ['SET'], 1.)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1.)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1.)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1.)

# Range of numbers, e.g. [1,100] (numbers 1 through 100)
grammar.add_rule('SET', 'range_set_', ['EXPR', 'EXPR', 'bound=100'], 10.)
# grammar.add_rule('SET', 'range_set_', ['1', '100', 'bound=100'], 10)
# Mapping expressions over sets of numbers
grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 3.)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR')

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], .1)

# Terminals
for i in range(0, len(INTEGERS)):
    grammar.add_rule('EXPR', str(INTEGERS[i]), None, TERMINAL_PRIOR * INTEGERS_DIST[i])

# lambda : mapset_(lambda y1: y1, mapset_(lambda y2: y2, setdifference_(mapset_(lambda y4: y4,
# union_(setdifference_(setdifference_(setdifference_(union_(intersection_(range_set_(5, 10, bound=100),
# range_set_(10, 1, bound=100)), range_set_(9, times_(4, 4), bound=100)), mapset_(lambda y9: y9,
# mapset_(lambda y10: 5, range_set_(7, 5, bound=100)))), range_set_(3, ipowf_(5, 2), bound=100)),
# range_set_(9, 9, bound=100)), range_set_(3, 6, bound=100))), range_set_(9, 6, bound=100))))
