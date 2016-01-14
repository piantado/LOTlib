
from math import log
from LOTlib.Eval import RecursionDepthException
from LOTlib.Miscellaneous import Infinity

class BinaryLikelihood(object):

    def compute_single_likelihood(self, datum):
        try:
            return log(datum.alpha * (self(*datum.input) == datum.output) + (1.0-datum.alpha) / 2.0)
        except RecursionDepthException as e: # we get this from recursing too deep -- catch and thus treat "ret" as None
            return -Infinity