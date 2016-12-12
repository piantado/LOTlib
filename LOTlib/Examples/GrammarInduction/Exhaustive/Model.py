from LOTlib.Examples.GrammarInduction.Data import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import LOTlib.Miscellaneous
from LOTlib.Miscellaneous import q

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

TERMINAL_WEIGHT = 2.

from LOTlib.Grammar import Grammar

## Here we use the _d primitives which manipulate an entire distribution of strings. This tends to be much faster.
grammar = Grammar()

grammar.add_rule('START', '', ['LIST'], 1.0)

grammar.add_rule('LIST', 'if_d', ['BOOL', 'LIST', 'LIST'], 1.)

grammar.add_rule('LIST', 'cons_d', ['ATOM', 'LIST'], 1./6.)
grammar.add_rule('LIST', 'cons_d', ['LIST', 'LIST'], 1./6.)
grammar.add_rule('LIST', 'cdr_d', ['LIST'], 1./3.)
grammar.add_rule('LIST', 'car_d', ['LIST'], 1./3.)

grammar.add_rule('LIST', '', ['ATOM'], 3.0)
grammar.add_rule('LIST', '{\'\':0.0}', None, 1.0)
# base_grammar.add_rule('LIST', 'recurse_', [], 1.) #

grammar.add_rule('BOOL', 'empty_d', ['LIST'], 1.)
grammar.add_rule('BOOL', 'flip_d(0.5)', None, 1.)

for t in 'DANV':
    grammar.add_rule('ATOM', '{\'%s\':0.0}' % t, None, 2.0)


## Allow lambda abstraction
grammar.add_rule('EXPR', 'apply_', ['LAMBDAARG', 'LAMBDATHUNK'], 1)
grammar.add_rule('LAMBDAARG',   'lambda', ['EXPR'], 1., bv_type='EXPR', bv_args=[] )
grammar.add_rule('LAMBDATHUNK', 'lambda', ['EXPR'], 1., bv_type=None, bv_args=None ) # A thunk

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLog
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

class MyHypothesis(MultinomialLikelihoodLog, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda : %s', **kwargs)
        self.outlier = -1000 # for MultinomialLikelihood

    def __call__(self, *args, **kwargs):
        # we have to mod this to insert the spaces since they aren't part of cons above
        ret = LOTHypothesis.__call__(self, *args, **kwargs)

        out = dict()
        for k,v in ret.items():
            out[' '.join(k)] = v
        return out

def make_hypothesis():
    return MyHypothesis(grammar)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)


