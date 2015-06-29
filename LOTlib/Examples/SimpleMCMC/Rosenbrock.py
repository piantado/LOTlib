
"""
Just playing around with vector-valued hypotheses. This is a simple sampler for a posterior shaped like
the exp(-RosenbrockFunction)

"""

import numpy

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import attrmem
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

class RosenbrockSampler(VectorHypothesis):

    def __init__(self, value=None):
        if value is None:
            value = numpy.array([0.0, 0.0])
        VectorHypothesis.__init__(self, value=value, n=2, proposal=numpy.eye(2)*0.1)

    """
    MCMC plays nicest if we have defined prior and likelihood, and just don't touch compute_posterior.

    """
    @attrmem('likelihood') # this makes sure the return values is memoized in self.likelihood
    def compute_likelihood(self, data, **kwargs):
        return 0.0 # just fixed to zero

    @attrmem('prior') # this makes sure the return values is memoized in self.prior
    def compute_prior(self):
        x,y = self.value
        return  -((1.0-x)**2.0 + 100.0*(y-x**2.0)**2.0)

    def propose(self):
        ## NOTE: Does not copy proposal
        newv = numpy.random.multivariate_normal(self.value, self.proposal)
        return RosenbrockSampler(value=newv), 0.0 # from symmetric proposals


if __name__ == "__main__":

    initial_hyp = RosenbrockSampler()

    for x in break_ctrlc(MHSampler(initial_hyp, [], 1000000, skip=100, trace=False)):
        print x, x.posterior_score
