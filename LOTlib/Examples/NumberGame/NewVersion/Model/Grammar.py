from LOTlib.Grammar import Grammar

grammar = Grammar()

# Default parameters for integer primitives
TERMINAL_PRIOR = 2.
INTEGERS = {
    1: 2.,
    2: 2.,
    3: 1.,
    4: 1.,
    5: 1.,
    # 6: .5,
    7: .5,
    # 8: .5,
    # 9: .5,
}

grammar.add_rule('START', '', ['SET'], 1.)

# Set theory
grammar.add_rule('SET', 'union_', ['SET', 'SET'], .5)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], .5)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], .5)

grammar.add_rule('SET', 'range_set_', ['EXPR', 'EXPR', 'bound=100'], 2.)
grammar.add_rule('SET', 'range_set_', ['1', '100', 'bound=100'], 10.)

# Mapping expressions over sets of numbers
grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 1.)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR', bv_p=2.)

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], .7)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], .7)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], .7)
grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], .7)

# Terminals
for i in INTEGERS.keys():
    grammar.add_rule('EXPR', str(i), None, TERMINAL_PRIOR * INTEGERS[i])
