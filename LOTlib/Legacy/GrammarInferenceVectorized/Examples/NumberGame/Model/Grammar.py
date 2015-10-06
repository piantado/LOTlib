"""

In the model used in Josh Tenenbaum's thesis, the following rules were used . . .

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
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar


# ============================================================================================================
# Mixture model grammar
# =====================
#
# This is identical to the model used in Josh Tenenbaum's thesis, except that (hopefully) we can do cooler
#  stuff with optimizing the hyperparameter lambda (aka `lambda_mix`) sampling GrammarHypotheses!
#
# Math rules & interval rules are the 2 probabilities mixed in this model.
#
#

mix_grammar = Grammar()
mix_grammar.add_rule('START', '', ['INTERVAL'], 1.)
mix_grammar.add_rule('START', '', ['MATH'], 1.)

mix_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
mix_grammar.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
mix_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)

# Math rules (30-40 of these)
# ---------------------------

# Odd numbers
mix_grammar.add_rule('EXPR', 'plus_', ['ODD', str(1)], 1.)
mix_grammar.add_rule('ODD', 'times_', ['X', str(2)], 1.)

# Primes
mix_grammar.add_rule('EXPR', 'isprime_', ['X'], 1.)

# Squares, cubes
mix_grammar.add_rule('EXPR', 'ipowf_', ['X', str(2)], 1.)
mix_grammar.add_rule('EXPR', 'ipowf_', ['X', str(3)], 1.)

# { 2^n  -  32 }
register_primitive(lambda x: x if x in (2, 4, 8, 16, 64) else 0, name='pow2n_d32_')
mix_grammar.add_rule('EXPR', 'pow2n_d32_', ['X'], 1.)
# { 2^n  &  37 }
register_primitive(lambda x: x if x in (2, 4, 8, 16, 32, 37, 64) else 0, name='pow2n_u37_')
mix_grammar.add_rule('EXPR', 'pow2n_u37_', ['X'], 1.)

# [2,12] * n
for i in range(2, 13):
    mix_grammar.add_rule('EXPR', 'times_', ['X', str(i)], 1.)

# [2,10] ^ m
for i in range(2, 11):
    mix_grammar.add_rule('EXPR', 'ipowf_', [str(i), 'X'], 1.)

# Ends in [0,9]
for i in range(0, 10):
    mix_grammar.add_rule('EXPR', 'ends_in_', ['X', str(i)], 1.)
    mix_grammar.add_rule('EXPR', 'contains_digit_', ['X', str(i)], 1.)

# Interval Rules (there will be ~5050 of these)
# ---------------------------------------------

mix_grammar.add_rule('INTERVAL', 'range_set_', ['CONST', 'CONST'], 1.)
for i in range(1, 101):
    mix_grammar.add_rule('CONST', '', [str(i)], 1.)


# ============================================================================================================
# independent-Priors Grammar
# =========================
#
# This has the same rules as the mixture model above, except each rule has an individual probability.
#
#  * With GrammarHypothesis, we can sample much more intricate models than the mixture model.
#  * However, we also will have like 5000 rules to choose from now . . .
#

independent_grammar = Grammar()

# Mixture params
# --------------
independent_grammar.add_rule('START', '', ['INTERVAL'], 1.)
independent_grammar.add_rule('START', '', ['MATH'], 1.)

# Math rules
# ----------
independent_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
independent_grammar.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
independent_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)

independent_grammar.add_rule('EXPR', 'plus_', ['ODD', str(1)], 1.)
independent_grammar.add_rule('ODD', 'times_', ['X', str(2)], 1.)
independent_grammar.add_rule('EXPR', 'isprime_', ['X'], 1.)
independent_grammar.add_rule('EXPR', 'ipowf_', ['X', str(2)], 1.)
independent_grammar.add_rule('EXPR', 'ipowf_', ['X', str(3)], 1.)
register_primitive(lambda x: x if x in (2, 4, 8, 16, 64) else 0, name='pow2n_d32_')
independent_grammar.add_rule('EXPR', 'pow2n_d32_', ['X'], 1.)
register_primitive(lambda x: x if x in (2, 4, 8, 16, 32, 37, 64) else 0, name='pow2n_u37_')
independent_grammar.add_rule('EXPR', 'pow2n_u37_', ['X'], 1.)
for i in range(2, 13):
    if not i==10:
        independent_grammar.add_rule('EXPR', 'times_', ['X', str(i)], 1.)
for i in range(2, 11):
    independent_grammar.add_rule('EXPR', 'ipowf_', [str(i), 'X'], 1.)
for i in range(0, 10):
    independent_grammar.add_rule('EXPR', 'ends_in_', ['X', str(i)], 1.)
    independent_grammar.add_rule('EXPR', 'contains_digit_', ['X', str(i)], 1.)

# Interval Rules
# --------------
independent_grammar.add_rule('INTERVAL', 'range_set_', ['CONST', 'CONST'], 1.)
for i in range(1, 101):
    independent_grammar.add_rule('CONST', '', [str(i)], 1.)


# ============================================================================================================
# LOTlib-Style Grammar
# ====================
#
# This grammar should generate all the hypotheses of the grammars listed above (and others!) as subset of
#  a larger language (possible w/ recursion). Basically, this grammar can recurse wherever there is an 'X',
#  and has leaves wherever there is a 'CONST'...
#
# We don't know what will happen so we can see, and compare it w/ our results above as well as Josh's
#  original results. Science!
#
# Note:
#   if we get rules like [X -> X*X] inflated to a high probability, we will probably get super-large
#   hypotheses that will break things
#
#

lot_grammar = Grammar()
lot_grammar.add_rule('START', '', ['SET'], 1.)

lot_grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], .1)
lot_grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], .1)
lot_grammar.add_rule('SET', 'union_', ['SET', 'SET'], .1)

lot_grammar.add_rule('SET', '', ['INTERVAL'], 1.)
lot_grammar.add_rule('SET', '', ['MATH'], 1.)

# Math rules
# ----------
lot_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'RANGE'], 1.)
lot_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'FULL_RANGE'], 1.)
lot_grammar.add_rule('FULL_RANGE', 'range_set_', ['1', '100'], 1.)
lot_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR', bv_p=2.)

lot_grammar.add_rule('EXPR', 'isprime_', ['EXPR'], 1.)
# NOTE: there is no distinction here between   2^n  &  n^2  !!!
lot_grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'EXPR'], .3)
lot_grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.)
lot_grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.)
lot_grammar.add_rule('EXPR', 'ends_in_', ['EXPR', 'EXPR'], 1.)
lot_grammar.add_rule('EXPR', 'contains_digit_', ['EXPR', 'EXPR'], 1.)

lot_grammar.add_rule('EXPR', '', ['OPCONST'], 20.)
for i in range(1, 11):
    lot_grammar.add_rule('OPCONST', '', [str(i)], 3.)
for i in [11, 12, 13, 14, 15]:
    lot_grammar.add_rule('OPCONST', '', [str(i)], 1.)

# Interval rules
# --------------
lot_grammar.add_rule('INTERVAL', '', ['RANGE'], 1.)
lot_grammar.add_rule('RANGE', 'range_set_', ['CONST', 'CONST'], 1.)
for i in range(1, 101):
    if i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        lot_grammar.add_rule('CONST', '', [str(i)], 5.)
    else:
        lot_grammar.add_rule('CONST', '', [str(i)], 1.)



import copy
import numpy as np

def grammar_gamma(grammar, scale=1.0):
    grammar = copy.copy(grammar)
    rules = [r for r in [r for sublist in grammar.rules.values() for r in sublist] if not (r.nt == 'CONST')]
    for r in rules:
        r.p = np.random.gamma(r.p, scale)
    return grammar








