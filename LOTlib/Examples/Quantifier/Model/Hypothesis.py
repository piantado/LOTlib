from copy import copy
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Lexicon.WeightedLexicon import WeightedLexicon
import Grammar as G


def make_my_hypothesis():
    return LOTHypothesis(G.grammar, args=['context'])

class GriceanQuantifierLexicon(WeightedLexicon):
    """A simple class that always fixes our generating function to LOTHypothesis.

        A weighted lexicon that evaluates the probability of a word depending on how likely it is to be used
        in a random (average) testing set.

    """
    def __init__(self, make_hypothesis, my_weight_function, alpha=0.9, palpha=0.9):
        WeightedLexicon.__init__(self, make_hypothesis, alpha=alpha, palpha=palpha)
        self.my_weight_function = my_weight_function

    def __copy__(self):
        new = type(self)(make_my_hypothesis(), self.my_weight_function, alpha=self.alpha, palpha=self.palpha)
        for w in self.value.keys():
            new.value[w] = copy(self.value[w])
        return new

    def weightfunction(self, u, context):
        return self.my_weight_function(self.value[u])


class MyContext(object):
    """Store a context."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)