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

# IF we want to define other constants:
#G.add_rule('EXPR', '', ['CONSTANT'], 10.0, resample_p=10000.0)

#G.add_rule('EXPR', 'PI', None, 1.0) # these terminals should have None for their function type; the literals
#G.add_rule('EXPR', 'E', None, 1.0) # these terminals should have None for their function type; the literals

#G.add_rule('CONSTANT', '0.0', None, 1.0)
#G.add_rule('CONSTANT', '2.0', None, 0.10)
#G.add_rule('CONSTANT', '3.0', None, 0.10)
#G.add_rule('CONSTANT', '4.0', None, 0.10)
#G.add_rule('CONSTANT', '5.0', None, 0.10)
#G.add_rule('CONSTANT', '6.0', None, 0.10)
#G.add_rule('CONSTANT', '7.0', None, 0.10)
#G.add_rule('CONSTANT', '8.0', None, 0.10)
#G.add_rule('CONSTANT', '9.0', None, 0.10)

#G.add_rule('CONSTANT', '0.01', None, 0.10)
#G.add_rule('CONSTANT', '0.10', None, 0.10)
#G.add_rule('CONSTANT', '10.0', None, 0.10)
#G.add_rule('CONSTANT', '100.0', None, 0.10)
#G.add_rule('CONSTANT', '1000.0', None, 0.10)
#G.add_rule('CONSTANT', '10000.0', None, 0.10)



	
	
	
	