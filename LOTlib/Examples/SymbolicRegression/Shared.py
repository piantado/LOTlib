# -*- coding: utf-8 -*-
"""
        Shared functions for symbolic regression.
"""
import LOTlib
from LOTlib.Grammar import Grammar
from LOTlib.BasicPrimitives import *
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *


#from SimpleMPI import MPI_map

from random import randint

## The grammar



#G.add_rule('CONSTANT', '', ['*gaussian*'], 10.0) ##TODO: HIGHLY EXPERIMENTAL
