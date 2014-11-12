from LOTlib.Grammar import Grammar

# Default parameters for integer primitives
TERMINAL_PRIOR = 5.
INTEGERS       = [1,   2,  3,  4,  5,  6,  7,  8,  9, 10]
INTEGERS_DIST  = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]   # len same as INT_PRIMITIVES


# Setting up our LOT hypothesis grammar
grammar = Grammar()

# Sets
grammar.add_rule('START', '', ['SET'], 1.)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1.)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1.)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1.)

# Range of numbers, e.g. [1,100] (numbers 1 through 100)
grammar.add_rule('SET', 'range_set_', ['EXPR', 'EXPR', 'bound=100'], 10.)
## grammar.add_rule('SET', '{1,2,3}', None, 10.)
## grammar.add_rule('SET', 'range_set_', ['1', '100', 'bound=100'], 10)

# Mapping expressions over sets of numbers
# grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 3.)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR')

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], .1)

# Terminals
for i in range(0, len(INTEGERS)):
    grammar.add_rule('EXPR', str(INTEGERS[i]), None, TERMINAL_PRIOR * INTEGERS_DIST[i])



# lambda : union_(setdifference_(range_set_(9, 1, bound=100), mapset_(lambda y3: 6, setdifference_(range_set_(1, 7, bound=100), setdifference_(range_set_(8, 3, bound=100), range_set_(7, 6, bound=100))))), range_set_(plus_(7, 5), times_(7, 3), bound=100))
# lambda : intersection_(range_set_(minus_(4, 5), 4, bound=100), mapset_(lambda y2: y2, mapset_(lambda y3: y3, setdifference_(intersection_(intersection_(range_set_(3, 10, bound=100), range_set_(times_(6, 8), 8, bound=100)), range_set_(7, 4, bound=100)), union_(range_set_(1, 5, bound=100), range_set_(9, 5, bound=100))))))
#
#

'''
intersection_(
    intersection_(setdifference_(
                      range_set_(6, plus_(minus_(1, 9), 1), bound=100),
                      range_set_(minus_(7, 3), 9, bound=100)),
                  range_set_(8, 7, bound=100)),
    intersection_(range_set_(2, 6, bound=100), range_set_(times_(1, 5), 4, bound=100)))
'''