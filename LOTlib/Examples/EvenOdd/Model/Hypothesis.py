from math import log
from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
from LOTlib.Evaluation.EvaluationException import RecursionDepthException

class EvenOddLexicon(RecursiveLexicon):

    def compute_single_likelihood(self, datum, ALPHA=0.9):
        ret = None
        try:  # Must catch errors here and NOT in __call__, since doing so in call will allow us to loop infinitely
            ret = self(*datum.input)
        except RecursionDepthException as e: # we get this from recursing too deep -- catch and thus treat "ret" as None
            pass

        return log(ALPHA*(ret == datum.output) + (1.-ALPHA)/2.)