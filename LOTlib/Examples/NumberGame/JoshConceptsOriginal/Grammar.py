from LOTlib.Grammar import Grammar

grammar = Grammar()

# Set theory
grammar.add_rule('START', '', ['SET'], 1.)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], .1)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], .1)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], .1)

grammar.add_rule('SET', 'range_set_', ['EXPR', 'EXPR', 'bound=100'], 10.)
grammar.add_rule('SET', 'range_set_', ['1', '100', 'bound=100'], 10)

# Mapping expressions over sets of numbers
grammar.add_rule('SET', 'mapset_', ['FUNC', 'SET'], 3.)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR', bv_p=3.5)

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], 3.)

# Terminals
for i in INTEGERS.keys():
    grammar.add_rule('EXPR', str(i), None, TERMINAL_PRIOR * INTEGERS[i])



lamb = 2./3.

grammar.add_rule('START', '', ['MATH'], lamb)
grammar.add_rule('START', '', ['INTERVAL'], 1-lamb)


'''
- even
- odd
- square
- *3
- *4
- *5
- *6
- *7
- *8
- *9
- *10
- end1
- end2
- end3
- end4
- end5
- end6
- end7
- end8
- end9
- ^2
- ^3
- ^4
- ^5
- ^6
- ^7
- ^8
- ^9
- ^10
- [1,100]
- 2^n, +37
- 2^n, -32
- primes
- cubes
- *11
- *12
-
-

5050:   all intervals 100 chose 2,  s.t. 1 <= n <= 100, n <= m <= 100  (incl. where n == m)

'''