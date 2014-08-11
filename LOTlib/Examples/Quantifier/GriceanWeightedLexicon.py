"""
        A weighted lexicon that evaluates the probability of a word depending on how likely it is to be used
        in a random (average) testing set
"""

from LOTlib.Miscellaneous import *
from LOTlib.Hypotheses.WeightedLexicon import WeightedLexicon
from copy import copy

class GriceanQuantifierLexicon(WeightedLexicon):
    """
            A simple class that always fixes our generating function to LOTHypothesis
    """

    def __init__(self, make_hypothesis, my_weight_function, alpha=0.9, palpha=0.9):
        WeightedLexicon.__init__(self, make_hypothesis, alpha=alpha, palpha=palpha)
        self.my_weight_function = my_weight_function

    def __copy__(self):
        new = type(self)(self.make_hypothesis, self.my_weight_function, alpha=self.alpha, palpha=self.palpha)
        for w in self.value.keys():
            new.value[w] = copy(self.value[w])
        return new

    def weightfunction(self, u, context):
        return self.my_weight_function(self.value[u])


def gricean_weight(h, testing_set, nu=1.0):
    """
    Takes a hypothesis and its function and returns the weight under a gricean setup, where the production probability is proportional to

    exp( 1.0 / (nu + proportionoftimeitistrue) )

    Note: The max weight is 1/nu, and this should not be huge compared to 1/alpha

    We (should) boundedly memoize this
    """

    pct = float(sum(map(lambda s: ifelse(h(s), 1.0, 0.0), testing_set) )) / len(testing_set)
    #pct = float(sum(map(lambda s: ifelse(f(*s) is True, 1.0, 0.0), testing_set) )) / len(testing_set) # pul out the context sets and apply f
    #pct = pct = float(sum(map(lambda s: ifelse(collapse_undef(f(*s)), 1.0, 0.0), testing_set) )) / len(testing_set) # pul out the context sets and apply f

    return 1.0 / (nu + pct)
