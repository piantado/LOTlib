"""
Vectorized GrammarHypothesis class.

This assumes:
    - domain hypothesis (e.g. NumberGameHypothesis) evaluates to a set
    - `data` is a list of FunctionData objects
    - input of FunctionData can be applied to hypothesis.compute_likelihood (for domain-level hypothesis)
    - output of FunctionData is a dictionary where each value is a pair (y, n),  where y is the number of
    positive and n is the number of negative responses for the given output key, conditioned on the input.

To use this for a different format, change init_R & compute_likelihood.

"""



# same for all GrammarHypotheses
# ------------------------------
#
# C = get rule counts for each grammar rule, for each hypothesis    |hypotheses| x |rules|
# for each FunctionData:
# Li = for FuncData i, for ea hypothesis, get likelihood of i.input in concept   |hypotheses| x 1
# Ri = for FuncData i, for ea hypothesis, is each i.output in the concept (1/0)  |hypotheses| x |output|


# compute_likelihood
# ------------------
#
# x = get rule probabilities for each rule    1 x |rules|
# P = x * C     |hypotheses| x 1
#
# for each FunctionData i:
#   v = Li + P      1 x |hypotheses|
#   Z = logsumexp(v)
#   v = exp(v-Z)        # weighted probability of each hypothesis given input data
#   p_in_concept = rowsum(v * Ri_j) for Ri_j in Ri   # i.e. multiply ea. col in Ri by v


import copy
from math import exp, log
import numpy as np
from numba import jit
from NumbaUtils import *
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Miscellaneous import gammaln

@jit
class GrammarHypothesisOptimized(GrammarHypothesis):

    @float_()
    def compute_likelihood(self, data, update_post=True, **kwargs):
        data_list
        return self.likelihood_optimized(data_list, update_post)

    @float_()
    def likelihood_optimized(self, data, update_post=True):
        """
        Compute the likelihood of producing human data, given:  H (self.hypotheses)  &  x (self.value)

        """
        # The following must be computed for this specific GrammarHypothesis
        # ------------------------------------------------------------------
        x = self.normalized_value()         # vector of rule probabilites
        P = np.dot(self.C, x)               # prior for each hypothesis
        likelihood = 0.0

        for d_key, d in enumerate(data):
            # Initialize unfilled values for L[data] & R[data]
            if d_key not in self.L:
                self.init_L(d, d_key)
            if d_key not in self.R:
                self.init_R(d, d_key)

            posteriors = self.L[d_key] + P
            Z = lse_numba(posteriors)
            w = posteriors - Z              # weights for each hypothesis

            # Compute likelihood of producing same output (yes/no) as data
            for m, o in enumerate(d.output.keys()):
                # col `m` of boolean matrix `R[i]` weighted by `w`
                p = calc_prob(w, self.R[d_key][:, m])
                # p = log((np.exp(w) * self.R[d_key][:, m]).sum())

                # NOTE: with really small grammars sometimes we get p > 0
                if p >= 0:
                    print 'P ERROR!'

                yes, no = d.output[o]
                k = yes
                n = yes + no
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp_numba(p)            # likelihood we got human output

        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    def normalized_value(self):
        """Return a rule probabilities, each normalized relative to other rules with same nt.

        Note
        ----
        This is the only time where we need to call `self.update()`, since this is the
        only time where we reference `self.rules`.

        """
        # self.update()

        # Make dictionary of normalization constants for each nonterminal
        nt_Z = {}
        for nt in self.grammar.nonterminals():
            Z = sum([self.value[i] for i in self.get_rules(rule_nt=nt)[0]])
            nt_Z[nt] = Z

        # Normalize each probability in `self.value`
        normalized = np.zeros(len(self.rules))
        for i, r in enumerate(self.rules):
            normalized[i] = self.value[i] / nt_Z[self.rules[i].nt]

        return np.log(normalized)

    def __copy__(self):
        return type(self)(
            self.grammar, self.hypotheses,
            H=self.H, C=self.C, L=self.L, R=self.R,
            value=copy.copy(self.value), proposal=self.proposal,
            prior_shape=self.prior_shape, prior_scale=self.prior_scale,
            propose_n=self.propose_n, propose_scale=self.propose_scale
        )

    def update(self):
        """
        Update `self.rules` relative to `self.value`.

        """
        # Set probability for each rule corresponding to value index
        for i in range(0, self.n):
            self.rules[i].p = self.value[i]
