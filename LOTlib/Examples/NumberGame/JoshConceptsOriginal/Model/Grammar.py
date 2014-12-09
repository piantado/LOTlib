
from LOTlib.Grammar import Grammar


def get_grammar(lambda_value=2./3.):
    grammar = Grammar()
    grammar.add_rule('START', 'in_domain_', ['MATH', '100'], lambda_value)
    grammar.add_rule('START', '', ['INTERVAL'], (1-lambda_value))

    '''
    Math rules (30-40 of these)

    '''
    grammar.add_rule('MATH', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
    grammar.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
    grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR', bv_p=1.)
    for i in range(2, 13):
        grammar.add_rule('EXPR', 'times_', ['EXPR', str(i)], 1.)
    for i in range(2, 11):
        grammar.add_rule('EXPR', 'ipowf_', ['EXPR', str(i)], 1.)
    for i in range(0, 10):
        grammar.add_rule('EXPR', 'ends_in_', ['EXPR', str(i)], 1.)

    '''
    Interval (there will be ~5050 of these)

    '''
    for n in range(1, 101):
        for m in range(n, 101):
            grammar.add_rule('INTERVAL', 'range_set_', [str(n), str(m)], 1.)

    return grammar




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
- *11
- *12
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


5050:   all intervals 100 chose 2,  s.t. 1 <= n <= 100, n <= m <= 100  (incl. where n == m)


TODO:
    - modulo / 'ends in n' primitive
    - is_prime primitive

'''




