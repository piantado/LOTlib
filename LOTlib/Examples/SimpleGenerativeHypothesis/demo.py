# -*- coding: utf-8 -*-

"""

         A demo of "syntax" learning using a SimpleGenerativeHypothesis.

         This searches over probabilistic generating functions, running them forward to estimate
         the likelihood of the data. Very very simple.
"""
import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import q
from LOTlib.Hypotheses.SimpleGenerativeHypothesis import SimpleGenerativeHypothesis

NDATA = 50 # How many of each data point have we seen?

data = {
        'N V'       : NDATA,
        'D N V'     : NDATA,
        'D N V N'   : NDATA,
        'D N V D N' : NDATA,
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Here is an example of using define_for_evaluator

from LOTlib.Evaluation.Eval import register_primitive # this creates new defines

# And this calls them:
register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
TERMINAL_WEIGHT = 5.

grammar = Grammar()
grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.)

grammar.add_rule('EXPR', 'if_', ['BOOL', 'EXPR', 'EXPR'], 1.)
grammar.add_rule('BOOL', 'equal_', ['EXPR', 'EXPR'], 1.)

grammar.add_rule('BOOL', 'flip_', [''], TERMINAL_WEIGHT)

# List-building operators
grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'cdr_', ['EXPR'], 1.)
grammar.add_rule('EXPR', 'car_', ['EXPR'], 1.)

grammar.add_rule('EXPR', '[]', None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('D'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('A'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('N'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('V'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('who'), None, TERMINAL_WEIGHT)


## Allow lambda abstraction
grammar.add_rule('EXPR', 'apply_', ['LAMBDAARG', 'LAMBDATHUNK'], 1)
grammar.add_rule('LAMBDAARG',   'lambda', ['EXPR'], 1., bv_type='EXPR', bv_args=[] )
grammar.add_rule('LAMBDATHUNK', 'lambda', ['EXPR'], 1., bv_type=None, bv_args=None ) # A thunk


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
from LOTlib.Proposals import *

h0 = SimpleGenerativeHypothesis(grammar )

## populate the finite sample by running the sampler for this many steps
for h in mh_sample(h0, data, 100000, skip=100):
    print h.posterior_score, h.prior, h.likelihood, h
    print h.llcounts
