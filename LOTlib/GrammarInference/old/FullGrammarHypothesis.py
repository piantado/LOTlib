import numpy as np
from random import random, randint
from copy import copy, deepcopy
from scipy.misc import logsumexp
from scipy.stats import dirichlet, binom, gamma, norm, beta
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Miscellaneous import sample1,self_update
from LOTlib.Hypotheses.Stochastics import *

class FullGrammarHypothesis(Hypothesis):
    """
    This hypothesis does inference over a grammar as well as parameters alpha (noise) and beta (base rate),
    and a likelihood temperature.
    """

    P_PROPOSE_RULEP = 0.75 # what proportion of the time do we propose to the rule probabilities?

    def __init__(self, Counts, L, GroupLength, prior_offset, Nyes, Ntrials, ModelResponse, value=None):
        """
            Counts - nonterminal -> #h x #rules counts
            L          - group -> #h x 1 array
            GroupLength   - #groups (vector)  - contains the number of trials per group
            Nyes          - #item ( #item = sum(GroupLength))
            Ntrials       - #item
            ModelResponse - #h x #item - each hypothesis' response to the i'th item (1 or 0)
        """

        assert sum(GroupLength) == len(Nyes) == len(Ntrials)

        L = numpy.array(L)

        self_update(self,locals())
        self.N_groups = len(GroupLength)
        self.nts    = Counts.keys() # all nonterminals
        self.nrules = { nt: Counts[nt].shape[1] for nt in self.nts} # number of rules for each nonterminal
        self.N_hyps = Counts[self.nts[0]].shape[0]

        if value is None:
            value = {
                      'rulep': { nt: DirichletDistribution(alpha=np.ones(self.nrules[nt]), proposal_scale=1000.) for nt in self.nts },
                      'alpha': BetaDistribution(1,1),
                      'beta':  BetaDistribution(1,1),
                      'likelihood_temperature':   GammaDistribution(a=1, scale=1, proposal_scale=10.),
                      'prior_temperature': GammaDistribution(a=1, scale=1, proposal_scale=10.)
            }

        Hypothesis.__init__(self, value=value) # sets the value

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        # The likelihood of the human data
        assert len(data) == 0

        alpha = self.value['alpha'].value[0]
        beta = self.value['beta'].value[0]
        llt = self.value['likelihood_temperature'].value
        pt = self.value['prior_temperature'].value

        # compute each hypothesis' prior, fixed over all data
        priors = np.ones(self.N_hyps) * self.prior_offset #   #h x 1 vector
        for nt in self.nts: # sum over all nonterminals
            priors = priors + np.dot(np.log(self.value['rulep'][nt].value), self.Counts[nt].T)

        priors = priors - np.log(sum(np.exp(priors)))
        priors = priors / pt # include prior temp

        pos = 0 # what response are we on?
        likelihood = 0.0
        # for g in [randint(0, self.N_groups - 1) for _ in xrange(10)]
        for g in xrange(self.N_groups):
            posteriors =  self.L[g]/llt + priors # posterior score
            posteriors = np.exp(posteriors - logsumexp(posteriors)) # posterior probability

            # Now compute the probability of the human data
            for _ in xrange(self.GroupLength[g]):
                ps = (1 - alpha) * beta + alpha * np.dot(posteriors, self.ModelResponse[pos])

                likelihood += binom.logpmf(self.Nyes[pos], self.Ntrials[pos], ps)
                pos = pos + 1

        return likelihood

    @attrmem('prior')
    def compute_prior(self):
        return self.value['alpha'].compute_prior() + \
               self.value['beta'].compute_prior() + \
               self.value['likelihood_temperature'].compute_prior() + \
               self.value['prior_temperature'].compute_prior() + \
               sum([ x.compute_prior() for x in self.value['rulep'].values()])


    def propose(self, epsilon=1e-10):
        # should return is f-b, proposal

        prop = type(self)(self.Counts, self.L, self.GroupLength, self.prior_offset, self.Nyes, \
                          self.Ntrials, self.ModelResponse, value=deepcopy(self.value))
        fb = 0.0

        if random() < FullGrammarHypothesis.P_PROPOSE_RULEP: # propose to the rule parameters

            nt = sample1(self.nts) # which do we propose to?

            prop.value['rulep'][nt], fb = prop.value['rulep'][nt].propose()

        else: # propose to one of the other grammar variables

            which = sample1(['alpha', 'beta', 'likelihood_temperature', 'prior_temperature'])

            prop.value[which], fb = prop.value[which].propose()

        return prop, fb