# same for all GrammarHypotheses
# ------------------------------
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


import copy
from math import exp, log
import numpy as np
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Miscellaneous import logsumexp, gammaln, log1mexp


class GrammarHypothesisVectorized(GrammarHypothesis):

    def __init__(self, grammar, hypotheses, H=None, C=None, L=None, R=None, **kwargs):
        GrammarHypothesis.__init__(self, grammar, hypotheses, **kwargs)
        if H is None:
            self.init_H()
        else:
            self.H = H
        if H is None:
            self.init_C()
        else:
            self.C = C
        self.L = L if L else {}
        self.R = R if R else {}

    def init_C(self):
        """
        Initialize our rule count vector `self.C`.

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

    def init_H(self):
        """
        Initialize hypothesis concept list `self.H`.

        """
        self.H = [h() for h in self.hypotheses]

    def init_L(self, d):
        """
        Initialize `self.L` dictionary.

        """
        self.L[d] = np.array([h.compute_likelihood(d.input) for h in self.hypotheses])  # For ea. hypo.

    def init_R(self, d):
        """
        Initialize `self.R` dictionary.

        """
        self.R[d] = np.zeros((len(self.hypotheses), len(d.output.keys())))
        for m, o in enumerate(d.output.keys()):
            self.R[d][:, m] = [int(o in h_concept) for h_concept in self.H]  # For ea. hypo.

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """
        Compute the likelihood of producing human data, given:  H (self.hypotheses)  &  x (self.value)

        """
        # Initialize unfilled values for L[data] & R[data]
        for d in data:
            if d not in self.L:
                self.init_L(d)
            if d not in self.R:
                self.init_R(d)

        # The following must be computed for this specific GrammarHypothesis
        # ------------------------------------------------------------------
        x = np.log(np.array(self.value))    # vector of rule probabilites
        P = np.dot(self.C, x)               # prior for each hypothesis
        likelihood = 0.0

        for d in data:
            posteriors = self.L[d] + P
            Z = logsumexp(posteriors)
            w = posteriors - Z              # weights for each hypothesis

            # Compute likelihood of producing same output (yes/no) as data
            for m, o in enumerate(d.output.keys()):
                # col `m` of boolean matrix `R[i]` weighted by `w`  -- TODO could this be logsumexp?
                p = log((np.exp(w) * self.R[d][:, m]).sum())
                p = -1e-10 if p >= 0 else p
                k = d.output[o][0]          # num. yes responses
                n = k + d.output[o][1]      # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    def __copy__(self):
        return type(self)(
            self.grammar, self.hypotheses,
            H=self.H, C=self.C, L=self.L, R=self.R,
            value=copy.copy(self.value), proposal=copy.copy(self.proposal),
            prior_shape=self.prior_shape, prior_scale=self.prior_scale,
            propose_n=self.propose_n, propose_step=self.propose_step
        )

    def update(self):
        """
        Update `self.rules` relative to `self.value`.

        """
        # Set probability for each rule corresponding to value index
        for i in range(1, self.n):
            self.rules[i].p = self.value[i]
