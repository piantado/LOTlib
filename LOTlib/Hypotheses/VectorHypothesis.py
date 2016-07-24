import copy
import random
import numpy as np
from Hypothesis import Hypothesis
from LOTlib.Miscellaneous import self_update


class VectorHypothesis(Hypothesis):
    """Store n-dimensional vectors (defaultly with Gaussian proposals)."""

    def __init__(self, value=None, n=1, proposal=None, propose_scale=1.0, propose_n=1):
        self.n = n
        self.propose_n = propose_n
        if value is None:
            value = np.random.multivariate_normal(np.array([0.0] * n), proposal)
        if proposal is None:
            proposal = np.eye(n) * propose_scale
        propose_mask = self.get_propose_mask()
        proposal = proposal * propose_mask
        self.proposal = proposal
        Hypothesis.__init__(self, value=value)
        self_update(self, locals())

    def propose(self):
        """New value is sampled from a normal centered @ old values, w/ proposal as covariance."""
        step = np.random.multivariate_normal(self.value, self.proposal)

        new_value = copy.copy(self.value)
        for i in random.sample(self.proposal.nonzero()[0], self.propose_n):
            new_value[i] = step[i]

        c = self.__copy__(new_value)
        return c, 0.0

    def get_propose_mask(self):
        """Default propose mask method."""
        return [True] * self.n

    def get_propose_idxs(self):
        return [i for i, m in enumerate(self.get_propose_mask()) if m]

    def compute_gradient(self, data, grad_step=.1):
        partials = np.zeros(self.n)
        posterior = self.compute_posterior(data)

        print '&'*110, ' GRADIENT'
        print 'POST: ', posterior

        for i in range(self.n):
            new_value = copy.copy(self.value)
            new_value[i] += grad_step
            c = self.__copy__(new_value)
            posterior_i = c.compute_posterior(data)
            partials[i] = (np.exp(posterior_i) - np.exp(posterior)) / grad_step
            print 'POST_I: ', posterior_i
        
        print '&'*110
        return partials

    def conditional_distribution(self, data, value_index, vals=np.arange(0, 2, .2)):
        """Compute posterior values for this grammar, varying specified value over a specified set.

        Args
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
            posterior = self.compute_posterior(data, updateflag=False)
            dist.append([self.prior, self.likelihood, posterior])

        self.set_value(old_value)
        return vals, dist

    def __copy__(self, value=None):
        """Copy this GH; shallow copies of value & proposal so we don't have sampling issues."""
        if value is None:
            value = copy.copy(self.value)
        proposal = copy.copy(self.proposal)
        c = type(self)()
        c.__dict__.update(self.__dict__)
        c.proposal = proposal
        c.set_value(value)
        return c

