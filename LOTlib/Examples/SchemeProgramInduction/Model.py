
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData

 # A little more interesting. Squaring: N parens go to N^2
#data = [
        #FunctionData( input=[[  [ [] ] * i  ]], output=[ [] ] * (i**2), alpha=alpha ) \
        #for i in xrange(1,10)
       #]

def make_data(alpha=0.99, size=1):
    # here just doubling x :-> cons(x,x)
    return [
        FunctionData(
            input=[[]],
            output=[[], []],
            alpha=alpha,
        ),
        FunctionData(
            input=[[[]]],
            output=[[[]], [[]]],
            alpha=alpha
        )
    ] * size

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A simple grammar for scheme, including lambda
grammar = Grammar()

# A very simple version of lambda calculus
grammar.add_rule('START', '', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'apply_', ['FUNC', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'x', None, 5.0)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1.0, bv_type='EXPR', bv_args=None)

grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'cdr_',  ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'car_',  ['EXPR'], 1.0)

grammar.add_rule('EXPR', '[]',  None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from math import log

class SchemeFunction(LOTHypothesis):

    # Prior, proposals, __init__ are all inherited from LOTHypothesis
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar, **kwargs)

    def compute_single_likelihood(self, datum):
        """
            Wrap in string for comparisons here. Also, this is a weird pseudo-likelihood (an outlier process)
            since when we are wrong, it should generate the observed data with some probability that's not going
            to be 1-ALPHA
        """
        if str(self(datum.input)) == str(datum.output):
            return log(datum.alpha)
        else:
            return log(1.0-datum.alpha)

def make_hypothesis(**kwargs):
    return SchemeFunction(**kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib import break_ctrlc

def scheme_generate():
    """ This generates random scheme code with cons, cdr, and car, and evaluates it on some simple list
    structures.

    No inference here -- just random sampling from a grammar.
    """

    example_input = [
        [],
        [[]],
        [[], []],
        [[[]]]
        ]

    ## Generate some and print out unique ones
    seen = set()
    for i in break_ctrlc(xrange(10000)):
        x = grammar.generate('START')

        if x not in seen:
            seen.add(x)

            # make the function node version
            f = LOTHypothesis(grammar, value=x, args=['x'])

            print x.log_probability(), x
            for ei in example_input:
                print "\t", ei, " -> ", f(ei)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)

