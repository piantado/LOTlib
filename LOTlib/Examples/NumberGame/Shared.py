__author__ = 'eric'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import Stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import sys
from math import log
from LOTlib import lot_iter
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import *
from LOTlib.Evaluation.Eval import *
from LOTlib.Miscellaneous import logsumexp, exp
from LOTlib.Inference.MetropolisHastings import MHSampler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Global values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INT_PRIMITIVES_PRIOR       = .25
INT_PRIMITIVES             = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
INT_PRIMITIVE_DISTRIBUTION = [1., 1., 1., .8, 1., .8, 1., .6, 1., .8]   # len same as INT_PRIMITIVES

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~ Setting up our LOT hypothesis grammar ~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
grammar = Grammar()

# Sets
grammar.add_rule('START', '', ['SET'], 1)
grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1)

# Range of numbers, e.g. [1,100] (numbers 1 through 100)
grammar.add_rule('SET', 'range_', ['EXPR', 'EXPR'], 5)
# Mapping expressions over sets of numbers
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1, bv_type='EXPR')
grammar.add_rule('SET', 'map_', ['FUNC', 'SET'], 1)

# Expressions
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], .1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], .1)

# Terminals
for i in range(0, len(INT_PRIMITIVES)):
    grammar.add_rule('EXPR', INT_PRIMITIVES[i], None,
                     INT_PRIMITIVES_PRIOR * INT_PRIMITIVE_DISTRIBUTION[i])


