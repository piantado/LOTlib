
from math import log
from LOTlib.Eval import TooBigException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis, Infinity
from LOTlib.Miscellaneous import attrmem

ALPHA = 0.95 # Default noise weight
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Primitives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class NumberGameHypothesis(LOTHypothesis):
    """
    Hypotheses evaluate to a subset of integers in [1, domain].
    """

    def __init__(self, grammar=None, value=None, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, value=value, display="lambda : %s", **kwargs)
        self.domain = domain

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability."""

        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            return -Infinity
        elif len(self()) == 0:
            return -Infinity
        else:
            # If all those checks pass, just return the tree log prob
            return self.grammar.log_probability(self.value) / self.prior_temperature

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        Args:
            data (FunctionData): this is the data; we only use data.input
            update_post (bool): boolean -- do we update posterior?

        """
        try:
            cached_set = self()      # Set of numbers corresponding to this hypothesis
        except OverflowError:
            cached_set = set()       # If our hypothesis call blows things up

        return sum([self.compute_single_likelihood(datum, cached_set) for datum in data]) / self.likelihood_temperature

    def compute_single_likelihood(self, d, cached_set=None):
        # the likelihood of getting all of these data points

        assert cached_set is not None, "*** We require precomputation of the hypothesis' set in compute_likelihood"
        assert len(d.input) == 0, "*** Required input is [] to use this implementation (functions are thunks)"

        ll = 0.0

        # Must sum over all elements in the set
        for di in d.output:
            if len(cached_set) > 0:
                ll += log(d.alpha*(di in cached_set)/len(cached_set) + (1.-d.alpha) / self.domain)
            else:
                ll += log( (1.-d.alpha) / self.domain)

        return ll

    def __call__(self, *args, **kwargs):
        # Sometimes self.value has too many nodes
        try:
            value_set = LOTHypothesis.__call__(self)
        except TooBigException:
            value_set = set()

        if isinstance(value_set, set):
            # Restrict our concept to being within our domain
            value_set = [x for x in value_set if (1 <= x <= self.domain)]
        else:
            # Sometimes self() returns None
            value_set = set()

        return value_set



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammars
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Eval import register_primitive

from LOTlib.Grammar import Grammar

# =======================
# Mixture model grammar
# =======================
"""
This implements the model from Josh Tenenbaum's thesis, the following rules were used:

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


# ===========================
# independent-Priors Grammar
# ===========================
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


# ====================
# LOTlib-Style Grammar
# ====================

lot_grammar = Grammar()
lot_grammar.add_rule('START', '', ['SET'], 1.)

lot_grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], .1)
lot_grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], .1)
lot_grammar.add_rule('SET', 'union_', ['SET', 'SET'], .1)

lot_grammar.add_rule('SET', '', ['RANGE'], 1.)
lot_grammar.add_rule('SET', '', ['MATH'], 1.)

# Math rules
# ----------
lot_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'RANGE'], 1.)
lot_grammar.add_rule('MATH', 'mapset_', ['FUNC', 'FULL_RANGE'], 1.)
lot_grammar.add_rule('FULL_RANGE', 'range_set_', ['1', '100'], 1.)
lot_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='EXPR', bv_p=2.)

lot_grammar.add_rule('EXPR', 'isprime_', ['EXPR'], 1.)
lot_grammar.add_rule('EXPR', 'ipowf_', ['EXPR', 'OPCONST'], .3)
lot_grammar.add_rule('EXPR', 'ipowf_', ['OPCONST', 'EXPR'], .3)
lot_grammar.add_rule('EXPR', 'times_', ['EXPR', 'OPCONST'], 1.)
lot_grammar.add_rule('EXPR', 'plus_', ['EXPR', 'OPCONST'], 1.)
lot_grammar.add_rule('EXPR', 'ends_in_', ['EXPR', 'OPCONST'], 1.)
lot_grammar.add_rule('EXPR', 'contains_digit_', ['EXPR', 'OPCONST'], 1.)

for i in range(1, 50):
    lot_grammar.add_rule('OPCONST', '', [str(i)], 1./i**2)

lot_grammar.add_rule('RANGE', 'range_set_', ['CONST', 'CONST'], 1.)
for i in range(1, 101):
    if i % 10 == 0: # decades get a higher prior
        lot_grammar.add_rule('CONST', '', [str(i)], 5.)
    else:
        lot_grammar.add_rule('CONST', '', [str(i)], 1.)









