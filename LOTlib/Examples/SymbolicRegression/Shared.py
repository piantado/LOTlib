# -*- coding: utf-8 -*-
"""
	Shared functions for symbolic regression. 
"""

from LOTlib.Grammar import Grammar
from LOTlib.BasicPrimitives import *
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis

#from SimpleMPI import MPI_map

from random import randint

## The grammar

G = Grammar()
G.add_rule('START', '', ['EXPR'], 1.0)

G.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)

G.add_rule('EXPR', 'exp_', ['EXPR'], 1.0)
G.add_rule('EXPR', 'log_', ['EXPR'], 1.0)
G.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 1.0) # including this gives lots of overflow

#G.add_rule('EXPR', 'sin_', ['EXPR'], 1.0)
#G.add_rule('EXPR', 'cos_', ['EXPR'], 1.0)
#G.add_rule('EXPR', 'tan_', ['EXPR'], 1.0)

G.add_rule('EXPR', 'x', None, 10.0) # these terminals should have None for their function type; the literals

G.add_rule('EXPR', '1.0', None, 100.0)

#G.add_rule('CONSTANT', '', ['*gaussian*'], 10.0) ##TODO: HIGHLY EXPERIMENTAL




	
	
	
	