import copy
import numpy
from Hypothesis import Hypothesis


class VectorHypothesis(Hypothesis):
    """Store n-dimensional vectors (defaultly with Gaussian proposals)."""

    def __init__(self, value=None, n=1, proposal=None):
        if proposal is None:
            proposal = numpy.eye(n)
        if value is None:
            value = numpy.random.multivariate_normal(numpy.array([0.0]*n), proposal)
        Hypothesis.__init__(self, value=value)
        self.n = n
        self.proposal = proposal
        self.__dict__.update(locals())

    def propose(self):
        """new value is sampled from a normal centered @ old values, w/ proposal as covariance (inverse?)"""
        newv = numpy.random.multivariate_normal(self.value, self.proposal)
        return type(self)(value=newv, n=self.n, proposal=self.proposal), 0.0

    def conditional_distribution(self, data, value_index, vals=numpy.arange(0, 2, .2)):
        """Compute posterior values for this grammar, varying specified value over a specified set.

        Args:
            data(list): List of datums.
            rule_name(string): Index of the value we're varying probabilities over.
            vals(list): List of float values.  E.g. [0,.2,.4, ..., 2.]

        Returns:
            list: List of [prior, likelihood, posterior], where each item corresponds to an item in the
            `vals` argument.

        """
        dist = []
        old_value = copy.copy(self.value)
        for p in vals:
            value = copy.copy(self.value)
            value[value_index] = p
            self.set_value(value)
            p, l = self.compute_posterior(data, updateflag=False)
            dist.append([p, l, p+l])

        self.set_value(old_value)
        return zip(vals, dist)

