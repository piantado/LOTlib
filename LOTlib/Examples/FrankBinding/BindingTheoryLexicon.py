from LOTlib.Hypotheses.SimpleLexicon import SimpleLexicon
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Evaluation.Primitives.Trees import tree_up_, is_nonterminal_type

from math import log
from copy import copy


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A class for binding theory hypotheses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class BindingTheoryLexicon(SimpleLexicon):
    """
        A version of StandardExpression that represents our binding theory hypotheses. It computes the likelihood
        by looping through the tree and finding how many examples are matched for each word.

        There is one option, <UNCHANGED>, which is the possibliity that we leave this expression unchanged
    """
    def __init__(self, make_hypothesis, words=(),  alpha=0.95, onlyNPs=True, **kwargs):
        self.onlyNPs = onlyNPs # do we only run this on NPs?
        SimpleLexicon.__init__(self, make_hypothesis, words, alpha=alpha, **kwargs)

    def compute_single_likelihood(self, datum):
        """
            onlyNPs - if True, we only apply the functions to NPs
        """
        assert len(datum.input)==1
        T = datum.input[0]
        Nwords = len(self.all_words())

        ll = 0.0
        for x in T:
            if self.onlyNPs and not is_nonterminal_type(x,"NP"): continue
            below = x.string_below()

            possible = [] # possible replacement words
            for w in self.all_words():

                if self.value[w](T,x): # if possible
                    if w == "<UNCHANGED>" and below not in self.value: w = below # have to give this as an option
                    possible.append(w) # the alternative word

            # compute the probability
            p = (1.0-self.alpha) / Nwords
            if len(possible)>0:
                p += self.alpha * (below in possible) / len(possible)

            ll += log(p)
        return ll















