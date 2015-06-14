from math import log
from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from Grammar import grammar, WORDS


class EvenOddLexicon(BinaryLikelihood, RecursiveLexicon):
    pass


def make_hypothesis():

    h = EvenOddLexicon()

    for w in WORDS:
        h.set_word(w, LOTHypothesis(grammar, args=['lexicon', 'x']))

    return h
