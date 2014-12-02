from LOTlib.Grammar import Grammar

grammar = Grammar()

# Default parameters for integer primitives
TERMINAL_PRIOR = 2.
INTEGERS = {
    1: 3.,
    2: 5.,
    3: .7,
    4: .6,
    5: 5.,
    # 6: .4,
    # 7: .3,
    # 8: .2,
    9: .1,
}

# Set theory
grammar.add_rule('START', '', ['SET'], 1.)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1.)
# grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], .1)
# grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], .1)

# grammar.add_rule('SET', 'range_set_', ['EXPR', 'EXPR', 'bound=100'], 10.)
grammar.add_rule('SET', 'range_set_', ['1', '100', 'bound=100'], 2)

# Mapping expressions over sets of numbers
grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 1.)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR')

# Expressions
# grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.)
# grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 10.)
grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], 10.)

# Terminals
for i in INTEGERS.keys():
    grammar.add_rule('EXPR', str(i), None, TERMINAL_PRIOR * INTEGERS[i])
