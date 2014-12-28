
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar


def mix_grammar(lambda_mix=2./3.):
    """
    Mixture model grammar with math rules & interval rules as the 2 probabilities mixed.

    Math Rules:  p = lambda
    -----------------------
    - primes
    - odd
    - even
    - squares
    - cubes
    - 2^n, +37
    - 2^n, -32
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
    - 2^n
    - 3^n
    - 4^n
    - 5^n
    - 6^n
    - 7^n
    - 8^n
    - 9^n
    - 10^n
    - range[1,100]

    Interval Rules:  p = (lambda - 1)
    ---------------------------------
    - all range[n,m] subset of r[1,100], such that n <=m  (5,050 rules like this!)


    """
    grammar = Grammar()
    # grammar.add_rule('START', 'in_domain_', ['MATH', '100'], lambda_value)  # TODO: is in_domain_ a prim??
    grammar.add_rule('START', '', ['MATH'], lambda_mix)
    grammar.add_rule('START', '', ['INTERVAL'], (1-lambda_mix))

    '''
    Math rules (30-40 of these)

    '''
    grammar.add_rule('MATH', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
    grammar.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
    grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)

    # Odd numbers
    grammar.add_rule('EXPR', 'plus_', ['ODD', str(1)], 1.)
    grammar.add_rule('ODD', 'times_', ['X', str(2)], 1.)
    # Primes
    grammar.add_rule('EXPR', 'isprime_', ['X'], 1.)
    # Squares, cubes
    grammar.add_rule('EXPR', 'ipowf_', ['X', str(2)], 1.)
    grammar.add_rule('EXPR', 'ipowf_', ['X', str(3)], 1.)
    # { 2^n  -  32 }
    register_primitive(lambda x: x if x in (2, 4, 8, 16, 64) else 0, name='pow2n_d32_')
    # { 2^n  &  37 }
    register_primitive(lambda x: x if x in (2, 4, 8, 16, 32, 37, 64) else 0, name='pow2n_u37_')


    for i in range(2, 13):
        grammar.add_rule('EXPR', 'times_', ['X', str(i)], 1.)
    for i in range(2, 11):
        grammar.add_rule('EXPR', 'ipowf_', [str(i), 'X'], 1.)
    for i in range(0, 10):
        grammar.add_rule('EXPR', 'ends_in_', ['X', str(i)], 1.)



    '''
    Interval (there will be ~5050 of these)

    '''
    for n in range(1, 101):
        for m in range(n, 101):
            grammar.add_rule('INTERVAL', 'range_set_', [str(n), str(m)], 1.)

    return grammar






