import numpy as np
from random import random
from copy import copy
from scipy.misc import logsumexp
from scipy.stats import dirichlet, binom, gamma, norm, beta
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Miscellaneous import sample1

class AlphaBetaGrammarHypothesis(Hypothesis):

    # Priors on parameters
    BETA_PRIOR = np.array([1.,1.])
    ALPHA_PRIOR = np.array([100., 100.])
    LLT_PRIOR = np.array([1., 1.])

    P_PROPOSE_VALUE = 0.25 # what proportion of the time do we propose to value?

    def __init__(self, Counts, Hypotheses, L, GroupLength, prior_offset, Nyes, Ntrials, ModelResponse, alpha=None, \
                 beta=None, llt=None, value=None,  scale=600, step_size=0.01):
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

        self.__dict__.update(locals())

        self.N_hyps = len(Hypotheses)
        self.N_groups = len(GroupLength)

        self.nts    = Counts.keys() # all nonterminals
        self.nrules = { nt: Counts[nt].shape[1] for nt in self.nts} # number of rules for each nonterminal

        # the dirichlet prior on parameters
        self.value_prior = { nt: np.ones(self.nrules[nt]) for nt in self.nts }
        self.set_value(value)
        Hypothesis.__init__(self)

    def set_value(self, value):
        # store the parameters in a hash from nonterminal to vector of probabilities
        if value is None:
            self.value = dict()
            for nt in self.nts:
                self.value[nt] = dirichlet.rvs(np.ones(self.nrules[nt]))[0] ## TODO: Check [0] here
        else:
            self.value = value

        if self.beta is None:
            self.beta = dirichlet.rvs(AlphaBetaGrammarHypothesis.BETA_PRIOR)[0][0]
        if self.alpha is None:
            self.alpha = dirichlet.rvs(AlphaBetaGrammarHypothesis.ALPHA_PRIOR)[0][0]
        if self.llt is None:
            self.llt = gamma.rvs(*AlphaBetaGrammarHypothesis.LLT_PRIOR, size=1)[0] # TODO: Check, parameters, rvs, size


    def compute_likelihood(self, data, **kwargs):
        # The likelihood of the human data
        assert len(data) == 0

        # compute each hypothesis' prior, fixed over all data
        priors = np.ones(self.N_hyps) * self.prior_offset #   #h x 1 vector
        for nt in self.nts: # sum over all nonterminals
            priors = priors + np.dot(np.log(self.value[nt]), self.Counts[nt].T) # TODO: Check .T

        pos = 0 # what response are we on?
        likelihood = 0.0
        for g in xrange(self.N_groups): ## TODO: Check offset
            posteriors =  self.L[g]/self.llt + priors # posterior score
            posteriors = np.exp(posteriors - logsumexp(posteriors)) # posterior probability

            # Now compute the probability of the human data
            for _ in xrange(1, self.GroupLength[g]):
                ps = (1 - self.alpha) * self.beta + self.alpha * np.dot(posteriors, self.ModelResponse[pos])
                # ps = np.dot(posteriors, self.ModelResponse[pos]) # model probabiltiy of saying yes # TODO: Check matrix multiply

                likelihood += binom.logpmf(self.Nyes[pos], self.Ntrials[pos], ps)
                pos = pos + 1

        self.likelihood = likelihood
        return likelihood

    def compute_prior(self):
        if self.alpha >= 1 or self.alpha <= 0:
            self.prior = -np.inf
            return self.prior
        else:
            self.prior = -1*(sum([ np.sum(dirichlet.logpdf(self.value[nt], self.value_prior[nt])) for nt in self.nts ]) + \
                                #dirichlet.logpdf(np.array([self.beta, 1.-self.beta]), AlphaBetaGrammarMH.BETA_PRIOR) + \
                                #dirichlet.logpdf(np.array([self.alpha, 1.-self.alpha]), AlphaBetaGrammarMH.ALPHA_PRIOR) + \
                                gamma.logpdf(self.llt, *AlphaBetaGrammarHypothesis.LLT_PRIOR)) # TODO: Check ordering of parameters
            return self.prior

    def propose(self, epsilon=1e-10):
        # should return is f-b, proposal

        if random() < AlphaBetaGrammarHypothesis.P_PROPOSE_VALUE:

            # uses epsilon smoothing to keep away from 0,1
            fb = 0.0

            # change value
            newvalue = dict()
            for nt in self.nts:
                inx = sample1(range(0, self.nrules[nt]))
                a = copy(self.value[nt])
                a[inx] = beta.rvs(self.value[nt][inx]*100, 100-self.value[nt][inx]*100) + epsilon

                #a = dirichlet.rvs(self.value[nt] * self.scale)[0] + epsilon
                newvalue[nt] = a / np.sum(a)
                fb += dirichlet.logpdf(newvalue[nt],self.value[nt]) - dirichlet.logpdf(self.value[nt],newvalue[nt])

            # make a new proposal. DON'T copy the matrices, but make a new value
            prop = AlphaBetaGrammarHypothesis(self.Counts, self.Hypotheses, self.L, self.GroupLength, self.prior_offset, self.Nyes,
                                      self.Ntrials, self.ModelResponse, value=newvalue, scale=self.scale, alpha=self.alpha,
                                      beta=self.beta, llt=self.llt)

            return prop, fb

        else:

            fb = 0.0

            newalpha = norm.rvs(loc=self.alpha, scale=self.step_size)
            fb += norm.logpdf(newalpha, self.alpha, self.step_size) - norm.logpdf(self.alpha, newalpha, self.step_size)

            newbeta = norm.rvs(loc=self.beta, scale=self.step_size) - 1e-5
            fb += norm.logpdf(newbeta, self.beta, self.step_size) - norm.logpdf(self.beta, newbeta, self.step_size)

            newllt = norm.rvs(loc=self.llt, scale=self.step_size)
            fb += norm.logpdf(newllt, self.llt, self.step_size) - norm.logpdf(self.llt, newllt, self.step_size)

            prop = AlphaBetaGrammarHypothesis(self.Counts, self.Hypotheses, self.L, self.GroupLength, self.prior_offset, self.Nyes,
                                      self.Ntrials, self.ModelResponse, value=self.value, scale=self.scale, alpha=newalpha,
                                      beta=newbeta, llt=newllt)

            return prop, fb