import numpy
from copy import copy
from scipy.stats import norm
from scipy.stats import dirichlet

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import Infinity, attrmem, logit, ilogit, sample1
from LOTlib.Hypotheses.Hypothesis import Hypothesis


class Stochastic(Hypothesis):
    """
    A Stochastic is a small class to allow MCMC on hypothesis parameters like temperature, noise, etc.
    It works as a Hypothesis, but is typically stored as a component of another hypothesis.
    """
    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=-Infinity, **kwargs):
        #raise NotImplementedError
        return 0.0

class NormalDistribution(Stochastic):

    def __init__(self, value=None, mean=0.0, sd=1.0, proposal_sd=1.0, **kwargs):
        Stochastic.__init__(self, value=value, **kwargs)
        self.mean = mean
        self.sd   = sd
        self.proposal_sd = proposal_sd

        if value is None:
            self.set_value(norm.rvs(loc=mean, scale=sd))

    @attrmem('prior')
    def compute_prior(self):
        return norm.logpdf(self.value, loc=self.mean, scale=self.sd)

    def propose(self):
        ret = copy(self)
        ret.value = norm.rvs(loc=self.value, scale=self.proposal_sd)

        return ret, 0.0 # symmetric

class LogitNormalDistribution(Stochastic):
    """
    Same as NormalDistribution, but value stores the logit value
    """

    def __init__(self, value=None, mean=0.0, sd=1.0, proposal_sd=1.0, **kwargs):
        Stochastic.__init__(self, value=value, **kwargs)
        self.mean = mean
        self.sd   = sd
        self.proposal_sd = proposal_sd

        if value is None:
            self.set_value(ilogit(norm.rvs(loc=mean, scale=sd)))

    @attrmem('prior')
    def compute_prior(self):
        return norm.logpdf(logit(self.value), loc=self.mean, scale=self.sd)

    def propose(self):
        ret = copy(self)
        ret.value = ilogit(norm.rvs(loc=logit(self.value), scale=self.proposal_sd))

        return ret, 0.0 # symmetric


class DirichletDistribution(Stochastic):

    SMOOTHING = 1e-6

    def __init__(self, value=None, alpha=None, proposal_scale=50.0, **kwargs):
        """
        Can be specified as value=numpy.array([...]), n= and alpha=
        """
        self.alpha = alpha

        if value is None and alpha is not None:
            value = numpy.random.dirichlet(alpha)

        Stochastic.__init__(self, value=value, **kwargs)

        self.proposal_scale = proposal_scale

    @attrmem('prior')
    def compute_prior(self):
        return dirichlet.logpdf(self.value, self.alpha)

    def propose(self):
        ret = copy(self)

        ret.value = numpy.random.dirichlet(self.value * self.proposal_scale)

        # add a tiny bit of smoothing away from 0/1
        ret.value = (1.0 - DirichletDistribution.SMOOTHING) * ret.value + DirichletDistribution.SMOOTHING / 2.0
        # and renormalize it, slightly breaking MCMC
        ret.value = ret.value / sum(ret.value)

        fb = dirichlet.logpdf(ret.value, self.value * self.proposal_scale) -\
             dirichlet.logpdf(self.value, ret.value * self.proposal_scale)

        return ret, fb

class GibbsDirchlet(DirichletDistribution):

    def propose(self):
        ret = copy(self)

        inx = sample1(range(0,self.alpha.shape[0]))
        ret.value[inx] = numpy.random.beta(self.value[inx]*self.proposal_scale,
                                           self.proposal_scale - self.value[inx] * self.proposal_scale)

        # add a tiny bit of smoothing away from 0/1
        ret.value[inx] = (1.0 - DirichletDistribution.SMOOTHING) * ret.value[inx] + DirichletDistribution.SMOOTHING / 2.0
        # and renormalize it, slightly breaking MCMC
        ret.value = ret.value / sum(ret.value)

        fb = dirichlet.logpdf(ret.value, self.value * self.proposal_scale) -\
             dirichlet.logpdf(self.value, ret.value * self.proposal_scale)

        return ret, fb

class BetaDistribution(DirichletDistribution):
    def __init__(self, a=1, b=1, **kwargs):
        DirichletDistribution.__init__(self, alpha=[a,b], **kwargs)



if __name__ == "__main__":
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

    h0 = BetaDistribution(1,2)
    for h in break_ctrlc(MHSampler(h0, [])):
        print h