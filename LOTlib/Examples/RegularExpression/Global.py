
"""
        Define a new kind of LOTHypothesis, that gives regex strings.
        These have a special interpretation function that compiles differently than straight python eval.
"""

from Data import *
from Grammar import *
from Specification import *

##########################################################
# make_h0
def make_h0(value=None):
    return RegexHypothesis(grammar, value=value, ALPHA=0.999)

