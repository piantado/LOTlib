
from math import log
from LOTlib.Eval import RecursionDepthException
from LOTlib.Miscellaneous import Infinity

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