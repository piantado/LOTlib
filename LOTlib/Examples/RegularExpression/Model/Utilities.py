
from LOTlib.Examples.RegularExpression.Model import *

def make_h0(value=None):
    """Define a new kind of LOTHypothesis, that gives regex strings.

    These have a special interpretation function that compiles differently than straight python eval.
    """
    return RegexHypothesis(grammar, value=value, ALPHA=0.999)

