"""
The more times we visit a hypothesis, the more we decrease its prior

TODO
----
* Try version where penalty decreases with time!
* This currently only extends LOTHypotheses, since we have to handle casting inside of h0 to WrapperClass.
  ... HOWEVER, we could make WrapperClass just dispatch the right methods if they don't exist

"""

from collections import Counter

from MetropolisHastings import MHSampler


class TabooMCMC(MHSampler):
    """
    An MCMC sampler that penalizes `self.posterior`

    Attributes
    ----------
    seen : Counter
        Keep track of all the samples we've drawn; this is a dictionary.
    penalty : float
        How much do we penalize for each sample?

    Note
    ----
    Requires storing of all hypotheses visited.

    """
    def __init__(self, h0, data, penalty=1.0, **kwargs):
        MHSampler.__init__(self, h0, data, **kwargs)
        self.penalty = penalty
        self.seen = Counter()

    def internal_sample(self, h):
        self.seen[h] += 1

    def compute_posterior(self, h, data):
        """
        Compute prior & likelihood for `h`, penalizing prior by how many samples have been generated so far.

        """
        return self.seen[h] * self.penalty + h.compute_posterior(data)


if __name__ == "__main__":

    from LOTlib.Examples.Number.Model import *
    from LOTlib.Miscellaneous import q

    data = generate_data(500)
    h0 = NumberExpression(grammar)
    for h in TabooMCMC(h0, data, steps=10000):

        print q(get_knower_pattern(h)), h.posterior_score, h.prior, h.likelihood, q(h)
