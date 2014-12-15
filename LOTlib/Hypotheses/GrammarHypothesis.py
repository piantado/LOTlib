
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

Methods:
    __init__
    propose
    compute_prior
    compute_likelihood
    rule_distribution
    set_value

    get_rules


"""
import copy
from math import exp, log
import random
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
    def __init__(self, grammar, hypotheses, value=None,
                 prior_shape=2., prior_scale=1., propose_n=1, propose_step=.1, **kwargs):
        self.grammar = grammar
        self.hypotheses = hypotheses
        self.rules = [r for sublist in grammar.rules.values() for r in sublist]
        if value is None:
            value = [rule.p for rule in self.rules]
        n = len(value)
        VectorHypothesis.__init__(self, value=value, n=n)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.propose_n = propose_n
        self.propose_step = propose_step
        self.propose_idxs = self.get_propose_idxs()
        self.prior = self.compute_prior()

    def propose(self):
        """Propose a new GrammarHypothesis; used to propose new samples with methods like MH.

        num_values => propose_n
        step_size => propose_step

        New value is sampled from a normal centered @ old values, w/ proposal as covariance (inverse?)

        """
        step = np.random.multivariate_normal([0.]*self.n, self.proposal) * self.propose_step
        c = self.__copy__()

        # randomly sample from our allowable indexes
        for i in random.sample(self.propose_idxs, self.propose_n):
            if c.value[i] + step[i] > 0.0:
                c.value[i] += step[i]

        c.set_value(c.value)
        return c, 0.0

    def compute_prior(self):
        shape = self.prior_shape
        scale = self.prior_scale
        rule_priors = [gamma.logpdf(v, shape, scale=scale) for v in self.value]

        prior = sum([r for r in rule_priors])
        self.prior = prior
        self.update_posterior()
        return prior

    def compute_likelihood(self, data, **kwargs):
        """Use hypotheses to estimate likelihood of generating the data.

        This is taken as a weighted sum over all hypotheses, sum { p(h | X) }

        Args:
            data(list): List of FunctionData objects.

        Returns:
            float: Likelihood summed over all outputs, summed over all hypotheses & weighted for each
            hypothesis by posterior score p(h|X).

        TODO:
            - vectorize

        """
        hypotheses = self.hypotheses
        likelihood = 0.0

        for d in data:
            posteriors = [sum(h.compute_posterior(d.input)) for h in hypotheses]
            Z = logsumexp(posteriors)
            weights = [(post-Z) for post in posteriors]

            for o in d.output.keys():
                # probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                # TODO: this will break if h() is None... can this ever happen??
                p = logsumexp([w if o in h() else -Infinity for h, w in zip(hypotheses, weights)])
                p = -1e-10 if p >= 0 else p      #
                k = d.output[o][0]         # num. yes responses
                n = k + d.output[o][1]     # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        self.likelihood = likelihood
        self.update_posterior()
        return likelihood

    def rule_distribution(self, data, rule_name, vals=np.arange(0, 2, .1)):
        """Compute posterior values for this grammar, varying specified rule over a set of values."""
        idxs, rules = self.get_rules(rule_name=rule_name)
        assert len(idxs) == 1, "\nERROR: More than 1 GrammarRule with this name!\n"

        return self.conditional_distribution(data, idxs[0], vals=vals)

    def set_value(self, value):
        """Set value and grammar rules for this hypothesis."""
        assert len(value) == len(self.rules), "ERROR: Invalid value vector!!!"
        self.value = value
        # Set probability for each rule corresponding to value index
        for i in range(1, len(value)):
            self.rules[i].p = value[i]

        # Recompute prior for each hypothesis, given new grammar probs
        for h in self.hypotheses:
            # re-set the tree generation_probabilities
            self.grammar.recompute_generation_probabilities(h.value)
            h.compute_prior()

    def get_rules(self, rule_name='XXX', rule_nt='XXX', rule_to='XXX'):
        """Get all GrammarRules associated with this rule name, 'nt' type, and/or 'to' types.

        Returns:
            Pair of lists [idxs, rules]: idxs is a list of rules indexes, rules is a list of GrammarRules

        """
        rules = [(i, r) for i, r in enumerate(self.rules)]
        if rule_name is not 'XXX':
            rules = [(i, r) for i, r in rules if r.name == rule_name]
        if rule_nt is not 'XXX':
            rules = [(i, r) for i, r in rules if r.nt == rule_nt]
        if rule_to is not 'XXX':
            rules = [(i, r) for i, r in rules if r.to == rule_to]
        return zip(*rules) if len(rules)>0 else [(), ()]

    def get_propose_idxs(self):
        """get list of indexes to alter -- we want to skip rules that have no alternatives/siblings"""
        proposal_indexes = range(self.n)
        nonterminals = self.grammar.nonterminals()
        for nt in nonterminals:
            idxs, r = self.get_rules(rule_nt=nt)
            if len(idxs) == 1:
                proposal_indexes.remove(idxs[0])
        return proposal_indexes

    def __copy__(self):
        """Make a shallow copy of this GrammarHypothesis."""
        return GrammarHypothesis(
            copy.copy(self.grammar), self.hypotheses,
            value=copy.copy(self.value), n=self.n, proposal=copy.copy(self.proposal),
            prior_shape=self.prior_shape, prior_scale=self.prior_scale,
            propose_n=self.propose_n, propose_step=self.propose_step
        )

    def print_best_hypotheses(self, n=10):
        hypotheses = self.hypotheses
        sorted_hypos = sorted(hypotheses, key=lambda x: x.posterior_score)
        for h in sorted_hypos[-n:]:
            print str(h)
            print h.posterior_score, h.likelihood, h.prior