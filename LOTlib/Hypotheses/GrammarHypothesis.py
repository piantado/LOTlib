
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

Let's say that we have a set of 'domain hypotheses', for example representing a concept in the number
domain such as 'powers of two between 1 and 100'. These 'domain hypotheses' are found in `self.hypotheses`.

Each GrammarHypothesis stands for a model of hyperparameters - a 'parameter hypothesis' - used to generate a
set of domain hypotheses.


Methods:
    __init__
    propose
    compute_prior
    compute_likelihood
    rule_distribution
    set_value

    get_rules

Note:
    Parts of this are currently designed just to work with NumberGameHypothesis... these should be expanded.

"""
import copy
from math import exp, log
import random
import pickle
import numpy as np
from scipy.stats import gamma
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity


class GrammarHypothesis(VectorHypothesis):
    """Hypothesis for grammar, we can represent grammar rule probability assignments as a vector.

    Inherits from VectorHypothesis, though I haven't figured out yet whe this really means...

    Attributes:
        grammar (LOTlib.Grammar): The grammar.
        hypotheses (LOTlib.Hypothesis): List of hypotheses, generated beforehand.
        rules (list): List of all rules in the grammar.
        value (list): Vector of numbers corresponding to the items in `rules`.

    """
    def __init__(self, grammar, hypotheses, rules=None, load=None, value=None,
                 prior_shape=2., prior_scale=1., propose_n=1, propose_step=.1,
                 **kwargs):
        self.grammar = grammar
        self.hypotheses = self.load_hypotheses(load) if load else hypotheses
        self.rules = [r for sublist in grammar.rules.values() for r in sublist]
        if value is None:
            value = [rule.p for rule in self.rules]
        n = len(value)
        VectorHypothesis.__init__(self, value=value, n=n, **kwargs)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        if int(self.n / 50) > propose_n:
            propose_n = int(self.n / 50)
        self.propose_n = propose_n
        self.propose_step = propose_step
        self.propose_idxs = self.get_propose_idxs()
        self.compute_prior()
        self.update()

    # --------------------------------------------------------------------------------------------------------
    # MCMC-Related methods

    def propose(self):
        """Propose a new GrammarHypothesis; used to propose new samples with methods like MH.

        New value is sampled from a normal centered @ old values, w/ proposal as covariance (inverse?)

        Note:
          * `self.propose_step` is used to determine how far to step with proposals.

        """
        step = np.random.multivariate_normal([0.]*self.n, self.proposal) * self.propose_step
        new_value = copy.copy(self.value)

        # randomly choose `self.propose_n` of our proposable indexes
        for i in random.sample(self.propose_idxs, self.propose_n):
            if new_value[i] + step[i] > 0.0:
                new_value[i] += step[i]

        c = self.__copy__()
        c.set_value(new_value)
        return c, 0.0

    def __copy__(self):
        """Copy of this GrammarHypothesis; `self.grammar` & `self.hypothesis` don't deep copy."""
        return type(self)(
            self.grammar, self.hypotheses,
            value=copy.copy(self.value), proposal=copy.copy(self.proposal),
            prior_shape=self.prior_shape, prior_scale=self.prior_scale,
            propose_n=self.propose_n, propose_step=self.propose_step
        )

    def set_value(self, value):
        """Set value and grammar rules for this hypothesis."""
        assert len(value) == len(self.rules), "ERROR: Invalid value vector!!!"
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.rules = [r for sublist in self.grammar.rules.values() for r in sublist]
        self.update()

    def update(self):
        """Update `self.grammar` & priors for `self.hypotheses` relative to `self.value`.

        We'll need to do this whenever we calculate things like predictive, because we need to use
          the `posterior_score` of each domain hypothesis get weights for our predictions.

        """
        # Set probability for each rule corresponding to value index
        for i in range(1, self.n):
            self.rules[i].p = self.value[i]

        # Recompute prior for each hypothesis, given new grammar probs
        for h in self.hypotheses:
            self.grammar.recompute_generation_probabilities(h.value)
            h.compute_prior()
            h.update_posterior()

    # --------------------------------------------------------------------------------------------------------
    # Bayesian inference with GrammarHypothesis

    def compute_prior(self):
        shape = self.prior_shape
        scale = self.prior_scale
        rule_priors = [gamma.logpdf(v, shape, scale=scale) for v in self.value]

        prior = sum([r for r in rule_priors])
        self.prior = prior
        self.update_posterior()
        return prior

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """Use bayesian model averaging with `self.hypotheses` to estimate likelihood of generating the data.

        This is taken as a weighted sum over all hypotheses, sum_h { p(h | X) } .

        Args:
            data(list): List of FunctionData objects.

        Returns:
            float: Likelihood summed over all outputs, summed over all hypotheses & weighted for each
            hypothesis by posterior score p(h|X).

        TODO:
            - vectorize

        """
        self.update()
        hypotheses = self.hypotheses
        likelihood = 0.0

        for d in data:
            posteriors = [sum(h.compute_posterior(d.input)) for h in hypotheses]
            Z = logsumexp(posteriors)
            weights = [(post-Z) for post in posteriors]

            for o in d.output.keys():
                C = h()
                assert C, "Error: hypothesis returned None!!"
                # probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                p = logsumexp([w if o in h() else -Infinity for h, w in zip(hypotheses, weights)])
                p = -1e-10 if p >= 0 else p
                k = d.output[o][0]         # num. yes responses
                n = k + d.output[o][1]     # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    # --------------------------------------------------------------------------------------------------------
    # p (y in C | H)  where H is our hypothesis space
    #
    # Note:
    #   This is NOT the same as `self.compute_likelihood` - that is a generative model, this is determining
    #   whether input would be part of our domain hypothesis concept(s).
    #

    def in_concept_mle(self, domain):
        """
        p(y in C | h_mle)   where h_mle is the max likelihood hypothesis

        """
        h_mle = self.get_top_hypotheses(n=1, key=(lambda x: x.likelihood))[0]
        C = h_mle()
        likelihoods = {}
        for y in domain:
            if y in C:
                likelihoods[y] = 1.
            else:
                likelihoods[y] = (h_mle.alpha - 1)
        return likelihoods

    def in_concept_map(self, domain):
        """
        p(y in C | h_map)   where h_map is the max a posteriori hypothesis

        """
        h_map = self.get_top_hypotheses(n=1, key=(lambda x: x.posterior_score))[0]
        C = h_map()
        likelihoods = {}
        for y in domain:
            if y in C:
                likelihoods[y] = 1.
            else:
                likelihoods[y] = (h_map.alpha - 1)
        return likelihoods

    # TODO is this right ?????????????
    def in_concept_avg(self, domain):
        """
        p(y in C | `self.hypotheses`)

        for each hypothesis h, if y in C_h, accumulated w_h where w is the weight of a hypothesis,
        determined by the hypothesis's posterior score p(h | y)

        ==> This is the weighted bayesian model averaging described in (Murphy, 2007)

        """
        self.update()
        probs_in_c = {}

        for y in domain:
            prob_in_c = 0
            Z = logsumexp([h.posterior_score for h in self.hypotheses])

            # for h in self.hypotheses:
            #     h.set_value(h.value)
            # print self.hypotheses[0].prior, self.hypotheses[3].prior, self.hypotheses[5].prior

            for h in self.hypotheses:
                C = h()
                w = h.posterior_score - Z
                if y in C:
                    prob_in_c += exp(w)
            probs_in_c[y] = prob_in_c

        return probs_in_c

    # --------------------------------------------------------------------------------------------------------
    # GrammarRule methods

    def get_rules(self, rule_name=False, rule_nt=False, rule_to=False):
        """Get all GrammarRules associated with this rule name, 'nt' type, and/or 'to' types.

        Note:
            rule_name is a string, rule_nt is a string, rule_to is a string, EVEN THOUGH rule.to is a list.

        Returns:
            Pair of lists [idxs, rules]: idxs is a list of rules indexes, rules is a list of GrammarRules

        """
        rules = [(i, r) for i, r in enumerate(self.rules) if r.nt is not 'IGNORE']

        # Filter our rules that don't match our criteria
        if rule_name is not False:
            rules = [(i, r) for i, r in rules if r.name == rule_name]
        if rule_nt is not False:
            rules = [(i, r) for i, r in rules if r.nt == rule_nt]
        if rule_to is not False:
            rules = [(i, r) for i, r in rules if rule_to in r.to]

        # Zip rules into separate `idxs` & `rules` lists
        return zip(*rules) if len(rules) > 0 else [(), ()]

    def get_propose_idxs(self):
        """Get indexes to propose to => only rules with siblings."""
        proposal_indexes = range(self.n)
        nonterminals = self.grammar.nonterminals()
        for nt in nonterminals:
            idxs, r = self.get_rules(rule_nt=nt)
            if len(idxs) == 1:
                proposal_indexes.remove(idxs[0])
        return proposal_indexes

    def rule_distribution(self, data, rule_name, vals=np.arange(0, 2, .1)):
        """Compute posterior values for this grammar, varying specified rule over a set of values."""
        idxs, rules = self.get_rules(rule_name=rule_name)
        assert len(idxs) == 1, "\nERROR: More than 1 GrammarRule with this name!\n"

        return self.conditional_distribution(data, idxs[0], vals=vals)

    # --------------------------------------------------------------------------------------------------------
    # Top domain hypotheses

    def get_top_hypotheses(self, n=10, key=(lambda x: x.posterior_score)):
        """Return the best `n` hypotheses from `self.hypotheses`."""
        self.update()
        hypotheses = self.hypotheses
        sorted_hypos = sorted(hypotheses, key=key)
        return sorted_hypos[-n:]

    def print_top_hypotheses(self, n=10):
        """Print the best `n` hypotheses from `self.hypotheses`."""
        self.update()
        for h in self.get_top_hypotheses(n=n):
            print str(h)
            print h.posterior_score, h.likelihood, h.prior

    def max_a_posteriori(self):
        self.update()
        return max([h.posterior_score for h in self.hypotheses])

    def max_like_estimate(self):
        self.update()
        return max([h.likelihood for h in self.hypotheses])

    # --------------------------------------------------------------------------------------------------------
    # Pickling domain hypotheses

    def load_hypotheses(self, filename=None):
        if filename:
            f = open(filename, "rb")
            self.hypotheses = pickle.load(f)
            return self.hypotheses

    def save_hypotheses(self, filename=None):
        if filename:
            f = open(filename, "wb")
            pickle.dump(self.hypotheses, f)

