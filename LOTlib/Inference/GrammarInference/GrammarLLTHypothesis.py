import numpy as np
from copy import deepcopy
from scipy.misc import logsumexp
from scipy.stats import binom
from LOTlib.Miscellaneous import sample1
from LOTlib.Hypotheses.Stochastics import *

class GrammarLLTHypothesis(Hypothesis):
    """
    This hypothesis does inference over a grammar as well as parameters alpha (noise) and beta (base rate),
    and a likelihood temperature.
    """

    def __init__(self, Counts, L, GroupLength, prior_offset, Nyes, Ntrials, ModelResponse, value=None):
        """
            Counts - nonterminal -> #h x #rules counts
            Hypotheses - #h
            L          - group -> #h x 1 array
            GroupLength   - #groups (vector)  - contains the number of trials per group
            Nyes          - #item ( #item = sum(GroupLength))
            Ntrials       - #item
            ModelResponse - #h x #item - each hypothesis' response to the i'th item (1 or 0)
        """

        assert sum(GroupLength) == len(Nyes) == len(Ntrials)

        L = numpy.array(L)

        self.__dict__.update(locals())
        self.N_groups = len(GroupLength)
        self.nts    = Counts.keys() # all nonterminals
        self.nrules = { nt: Counts[nt].shape[1] for nt in self.nts} # number of rules for each nonterminal
        self.N_hyps = Counts[self.nts[0]].shape[0]

        if value is None:
            value = {
                      'rulep': { nt: GibbsDirchlet(alpha=np.ones(self.nrules[nt]), proposal_scale=100.) for nt in self.nts },
                      'llt':   NormalDistribution(1,0.1) # TODO: Should be a lognormal or gamma
                     }

        Hypothesis.__init__(self, value=value) # sets the value

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        # The likelihood of the human data
        assert len(data) == 0

        # compute each hypothesis' prior, fixed over all data
        priors = np.ones(self.N_hyps) * self.prior_offset #   #h x 1 vector
        for nt in self.nts: # sum over all nonterminals
            priors = priors + np.dot(np.log(self.value['rulep'][nt].value), self.Counts[nt].T)

        llt   = abs(self.value['llt'].value)

        pos = 0 # what response are we on?
        likelihood = 0.0
        for g in xrange(self.N_groups): ## TODO: Check offset
            posteriors =  self.L[g] / llt + priors # posterior score
            posteriors = np.exp(posteriors - logsumexp(posteriors)) # posterior probability

            # Now compute the probability of the human data
            for _ in xrange(1, self.GroupLength[g]):
                ps = np.dot(posteriors, self.ModelResponse[pos])

                likelihood += binom.logpmf(self.Nyes[pos], self.Ntrials[pos], ps)
                pos = pos + 1

        return likelihood

    @attrmem('prior')
    def compute_prior(self):
        return self.value['llt'].compute_prior() + \
               sum([ x.compute_prior() for x in self.value['rulep'].values()])


    def propose(self, epsilon=1e-10):
        # should return is f-b, proposal

        prop = type(self)(self.Counts, self.L, self.GroupLength, self.prior_offset, self.Nyes, \
                          self.Ntrials, self.ModelResponse, value=deepcopy(self.value))

        nt = sample1(self.nts) # which do we propose to?

        prop.value['rulep'][nt], fb0 = prop.value['rulep'][nt].propose()
        prop.value['llt'], fb1 = prop.value['llt'].propose()

        return prop, fb0+fb1
