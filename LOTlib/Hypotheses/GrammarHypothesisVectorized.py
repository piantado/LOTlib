# initialization
# --------------
#
# C = get rule counts for each grammar rule, for each hypothesis    |hypotheses| x |rules|
# for each FunctionData:
# Li = for FuncData i, for ea hypothesis, get likelihood of ea. input in concept   |hypotheses| x 1
# Ri = for FuncData i, for ea hypothesis, is each output in the concept (1/0)  |hypotheses| x |output|


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


from math import exp, log
import numpy as np
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Miscellaneous import logsumexp, gammaln, log1mexp


class GrammarHypothesisVectorized(GrammarHypothesis):

    def initialize_vector(self, data):
        """
        Initialize our rule count & domain-hypothesis-likelihood vectors (C, L, R).

        """
        self.C = np.zeros((len(self.hypotheses), len(self.rules)))
        rule_idxs = {r: i for i, r in enumerate(self.rules)}
        for j, h in enumerate(self.hypotheses):
            grammar_rules = [fn.rule for fn in h.value.subnodes()[1:]]
            for rule in grammar_rules:
                try:
                    self.C[j, rule_idxs[rule]] += 1
                except Exception:
                    if isinstance(rule, BVUseGrammarRule):
                        pass
                    else:
                        raise Exception

        self.L = [None] * len(data)
        self.R = [None] * len(data)
        for i, d in enumerate(data):
            self.L[i] = np.array([h.compute_likelihood(d.input) for h in self.hypotheses])  # For ea. hypo.
            self.R[i] = np.zeros((len(self.hypotheses), len(d.output.keys())))
            # For each Output data point...
            for m, o in enumerate(d.output.keys()):
                self.R[i][:, m] = [int(o in h()) for h in self.hypotheses]  # For ea. hypo.

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """
        Compute the likelihood of producing human data, given:  H (self.hypotheses)  &  x (self.value)

        """
        try:                        # only call `self.initialize_vector` the first time we compute likelihood
            if self.C:
                pass
        except Exception:
            self.initialize_vector(data)

        x = np.log(np.array(self.value))    # vector of rule probabilites
        P = np.dot(self.C, x)               # prior for each hypothesis
        # print 'x', x
        # print 'C', self.C
        # print 'P', P
        likelihood = 0.0

        for i, d in enumerate(data):
            posteriors = self.L[i] + P
            Z = logsumexp(posteriors)
            w = posteriors - Z              # weights for each hypothesis

            # Compute likelihood of producing same output (yes/no) as data
            for m, o in enumerate(d.output.keys()):
                # p = logsumexp(w * self.R[i][:, m])      # col `m` of boolean matrix `R[i]` weighted by `w`
                p = log(sum(np.exp(w) * self.R[i][:, m]))

                # print 'w', w
                # print 'p', p
                p = -1e-10 if p >= 0 else p
                k = d.output[o][0]         # num. yes responses
                n = k + d.output[o][1]     # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

