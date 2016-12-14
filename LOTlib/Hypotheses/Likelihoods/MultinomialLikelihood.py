
from math import log
from LOTlib.Eval import RecursionDepthException
from LOTlib.Miscellaneous import Infinity, logsumexp

class MultinomialLikelihood(object):
    """
    Compute multinomial likelihood for data where the data is a dictionary and the function
    output is also a dictionary of counts. The self dictionary gets normalized to be a probability
    distribution. Smoothing term called "outlier" is the (unnormalized) probability assigned to
    out of sample items
    """

    def compute_single_likelihood(self, datum):
        outlier = self.__dict__.get('outlier', -Infinity) # pull the outleir prob from self adjective

        assert isinstance(datum.output, dict)

        hp = self(*datum.input) # output dictionary, output->probabilities
        assert isinstance(hp, dict)
        try:
            return sum( dc * (log(hp[k]) if k in hp else outlier) for k, dc in datum.output.items() )
        except ValueError as e:
            print "*** Math domain error", hp, str(self)
            raise e

class MultinomialLikelihoodLog(object):
    """
    Same but assumes we get out log likelihoods
    """

    def compute_single_likelihood(self, datum):
        outlier = self.__dict__.get('outlier', -Infinity)  # pull the outleir prob from self adjective

        assert isinstance(datum.output, dict)

        hp = self(*datum.input)  # output dictionary, output->probabilities
        assert isinstance(hp, dict)
        try:
            return sum(dc * (hp[k] if k in hp else outlier) for k, dc in datum.output.items())
        except ValueError as e:
            print "*** Math domain error", hp, str(self)
            raise e

from Levenshtein import distance

class MultinomialLikelihoodLogLevenshtein(object):
    """
    Assume a levenshtein edit distance on strings. Slower, but more gradient. Not a real likelihood
    """

    def compute_single_likelihood(self, datum):
        distance_scale = self.__dict__.get('distance', 1.0)

        assert isinstance(datum.output, dict)

        hp = self(*datum.input)  # output dictionary, output->probabilities
        assert isinstance(hp, dict)
        try:

            # now we have to add up every string that we could get
            return sum(dc * ( logsumexp([rlp - distance_scale*distance(r, k) for r, rlp in hp.items()]))\
                           for k, dc in datum.output.items())

        except ValueError as e:
            print "*** Math domain error", hp, str(self)
            raise e

def prefix_distance(x,s):
    """
    if x is a prefix of s, return the number of chars remaining in s
    otherwise, return -Infinity.
    This is used as a distance metric that prefers to get the beginning of strings right, to allow
    us to model deeper program search as covering more and more of the prefix of a string.
    Interestingly, it doesn't allow x to be longer than s, meaning we really care about getting s right
    """

    if len(x) > len(s): # x cannot be a prefix of s
        return Infinity
    elif s[:len(x)] == x:
        return len(s)-len(x)
    else:
        return Infinity


class MultinomialLikelihoodLogPrefixDistance(object):
    """
    This distance between strings here is the remainder
    """

    def compute_single_likelihood(self, datum):
        distance_scale = self.__dict__.get('distance', 1.0)

        assert isinstance(datum.output, dict)

        hp = self(*datum.input)  # output dictionary, output->probabilities
        assert isinstance(hp, dict)
        try:

            # now we have to add up every string that we could get
            return sum(dc * ( logsumexp([rlp - distance_scale*prefix_distance(r, k) for r, rlp in hp.items()]))\
                           for k, dc in datum.output.items())

        except ValueError as e:
            print "*** Math domain error", hp, str(self)
            raise e