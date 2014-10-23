"""
        The more times we visit a hypothesis, the more we decrease its prior

        TODO: Try version where penalty decreases with time!
        TODO: This currently only extends LOTHypotheses, since we have to handle casting
              inside of h0 to WrapperClass. HOWEVER, we could make WrapperClass just dispatch the right methods
              if they don't exist
"""

from MetropolisHastings import MHSampler
from collections import Counter

class TabooMCMC(MHSampler):
    """
            An MCMC sampler that penalizes for visits to a hypothesis
            NOTE: rEquires storing of all hypotheses visited.
    """
    def __init__(self, h0, data, penalty=1.0, **kwargs ):
        MHSampler.__init__(self, h0, data, **kwargs)
        self.penalty=penalty

        self.seen = Counter()

    def internal_sample(self, h):
        """
                Keep track of how many samples we've drawn for h
        """
        self.seen[h] += 1

    def compute_posterior(self, h, data):
        """
                Wrap the posterior with a penalty for how often we've seen h. Computes the penalty on the prior
        """
        mypenalty = self.seen[h] * self.penalty
        np, nl = MHSampler.compute_posterior(self, h, data)
        return np+mypenalty, nl


if __name__ == "__main__":

    from LOTlib.Examples.Number.Model.Inference import generate_data, NumberExpression, grammar, get_knower_pattern
    from LOTlib.Miscellaneous import q

    data = generate_data(500)
    h0 = NumberExpression(grammar)
    for h in TabooMCMC(h0, data, steps=10000):

        print q(get_knower_pattern(h)), h.posterior_score, h.prior, h.likelihood, q(h)
