

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Hypothesis import EvenOddLexicon
from Grammar import grammar, WORDS

def make_meaning():
    """
    The form a single word takes. Must have "lexicon" as its first argument to be used in a
    RecursiveLexicon
    """
    return LOTHypothesis(grammar, args=['lexicon', 'x'])

def make_h0():
    return EvenOddLexicon(make_meaning, words=WORDS)


