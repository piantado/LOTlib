from math import log
from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
from LOTlib.Evaluation.EvaluationException import RecursionDepthException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar, WORDS

def make_meaning():
    """
    The form a single word takes. Must have "lexicon" as its first argument to be used in a
    RecursiveLexicon
    """
    return LOTHypothesis(grammar, args=['lexicon', 'x'])


class EvenOddLexicon(RecursiveLexicon):

    def compute_single_likelihood(self, datum, ALPHA=0.9):
        ret = None
        try:  # Must catch errors here and NOT in __call__, since doing so in call will allow us to loop infinitely
            ret = self(*datum.input)
        except RecursionDepthException as e: # we get this from recursing too deep -- catch and thus treat "ret" as None
            pass

        return log(ALPHA*(ret == datum.output) + (1.-ALPHA)/2.)


def make_hypothesis():
    return EvenOddLexicon(make_meaning, words=WORDS)