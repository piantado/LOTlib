from LOTlib.DataAndObjects import FunctionData
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from math import log


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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A class for scheme hypotheses that just computes the input/output pairs with the appropriate probability
class SchemeFunction(LOTHypothesis):

    # Prior, proposals, __init__ are all inherited from LOTHypothesis

    def compute_single_likelihood(self, datum):
        """
                Wrap in string for comparisons here. Also, this is a weird pseudo-likelihood (an outlier process)
                since when we are wrong, it should generate the observed data with some probability that's not going
                to be 1-ALPHA
        """

        if str(self(datum.input)) == str(datum.output):
            return log(self.ALPHA)
        else:
            return log(1.0-self.ALPHA)
