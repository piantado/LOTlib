# -*- coding: utf-8 -*-
"""
Functions for gricean.

"""
from LOTlib import lot_iter
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *
from LOTlib.FunctionNode import FunctionNode
from LOTlib.FiniteBestSet import FiniteBestSet

from random import randint
from copy import copy

from GriceanWeightedLexicon import gricean_weight

import Hypothesis as H



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_my_hypothesis():
    return LOTHypothesis(grammar, args=['context'])

from cachetools import lru_cache

@lru_cache
def my_weight_function(h):
    return gricean_weight(h, TESTING_SET)
