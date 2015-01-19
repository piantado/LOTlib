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

# ------------------------------------------------------------------------------------------------------------
# Math rules (30-40 of these)

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

# ------------------------------------------------------------------------------------------------------------
# Interval Rules (there will be ~5050 of these)

for n in range(1, 101):
    for m in range(n, 101):
        mix_grammar.add_rule('INTERVAL', 'range_set_', [str(n), str(m)], 1.)


# ============================================================================================================
# Individual-Priors Grammar
# =========================
#
# This has the same rules as the mixture model above, except each rule has an individual probability.
#
#  * With GrammarHypothesis, we can sample much more intricate models than the mixture model.
#  * However, we also will have like 5000 rules to choose from now . . .
#
#
individual_grammar = Grammar()

# ------------------------------------------------------------------------------------------------------------
# Math rules

individual_grammar.add_rule('START', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
individual_grammar.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
individual_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)

individual_grammar.add_rule('EXPR', 'plus_', ['ODD', str(1)], 1.)
individual_grammar.add_rule('ODD', 'times_', ['X', str(2)], 1.)
individual_grammar.add_rule('EXPR', 'isprime_', ['X'], 1.)
individual_grammar.add_rule('EXPR', 'ipowf_', ['X', str(2)], 1.)
individual_grammar.add_rule('EXPR', 'ipowf_', ['X', str(3)], 1.)
register_primitive(lambda x: x if x in (2, 4, 8, 16, 64) else 0, name='pow2n_d32_')
individual_grammar.add_rule('EXPR', 'pow2n_d32_', ['X'], 1.)
register_primitive(lambda x: x if x in (2, 4, 8, 16, 32, 37, 64) else 0, name='pow2n_u37_')
individual_grammar.add_rule('EXPR', 'pow2n_u37_', ['X'], 1.)
for i in range(2, 13):
    individual_grammar.add_rule('EXPR', 'times_', ['X', str(i)], 1.)
for i in range(2, 11):
    individual_grammar.add_rule('EXPR', 'ipowf_', [str(i), 'X'], 1.)
for i in range(0, 10):
    individual_grammar.add_rule('EXPR', 'ends_in_', ['X', str(i)], 1.)

# ------------------------------------------------------------------------------------------------------------
# Interval Rules

individual_grammar.add_rule('START', 'range_set_', ['CONST', 'CONST'], 1.)
for i in range(1, 101):
    individual_grammar.add_rule('CONST', '', [str(i)], 1.)


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

# Initial range stuff -- note that we have a mixture model with range[1,100] & range[CONST,CONST],
#  where CONST is the same constant atom used in the math expressions below.
lot_grammar.add_rule('START', 'mapset_', ['FUNC', 'RANGE'], 1.)
lot_grammar.add_rule('RANGE', 'range_set_', ['RANGE_CONST', 'RANGE_CONST'], 1.)
lot_grammar.add_rule('RANGE', 'range_set_', ['1', '100'], 1.)
lot_grammar.add_rule('FUNC', 'lambda', ['X'], 1., bv_type='X', bv_p=1.)
for i in range(1, 101):
    lot_grammar.add_rule('RANGE_CONST', '', [str(i)], 1.)

# Math expressions
lot_grammar.add_rule('X', 'isprime_', ['X'], 1.)
lot_grammar.add_rule('X', 'ipowf_', ['CONST', 'CONST'], 1.)
lot_grammar.add_rule('X', 'ipowf_', ['X', 'CONST'], 1.)
lot_grammar.add_rule('X', 'ipowf_', ['CONST', 'X'], 1.)
lot_grammar.add_rule('X', 'ipowf_', ['X', 'X'], 1.)
lot_grammar.add_rule('X', 'times_', ['CONST', 'CONST'], 1.)
lot_grammar.add_rule('X', 'times_', ['X', 'CONST'], 1.)
lot_grammar.add_rule('X', 'times_', ['X', 'X'], 1.)
lot_grammar.add_rule('X', 'plus_', ['CONST', 'CONST'], 1.)
lot_grammar.add_rule('X', 'plus_', ['X', 'CONST'], 1.)
lot_grammar.add_rule('X', 'plus_', ['X', 'X'], 1.)
lot_grammar.add_rule('X', 'ends_in_', ['X', 'CONST'], 1.)

# Constants
for i in range(1, 11):
    lot_grammar.add_rule('CONST', '', [str(i)], 1.)



# TRY: range 'START' & range 'END'
# TRY: keep 'CONST' or something like it, but tell GrammarHypothesis to ignore this?  ==> 'IGNORE' nt
#
# TRY: another thing to do would be playing with greater values in propose_n... with 5000 rules,
#      we need to propose to more than 1 value at a time

# bayesian data analysis  +  probabilistic, structured LOT model
# fitting priors in LOT models... what's new is we're doing a BDA that can recover plausible priors for LOT
#  models

# theres other work in psychophysics that tries to recover / infer priors (things like 'what are your
# priors on direction of motion' or what are your priors on speed?') draws on classic structured AI
# approaches, combined with cool data analysis that can pull out the priors
# ==> from here, you can do cool things with these structured models to suppose whats really happening


