
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

Methods:
    __init__
    propose
    compute_prior
    compute_likelihood
    rule_distribution
    set_value

    get_rule
    get_rules
    get_rule_by_index
    get_rule_index


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
    def __init__(self, grammar, hypotheses, value=None, prior_shape=2., prior_scale=1.,
                 propose_n=1, propose_step=.1, **kwargs):
        self.rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        self.grammar = grammar
        self.hypotheses = hypotheses
        if value is None:
            value = [rule.p for rule in self.rules]
        n = len(value)
        VectorHypothesis.__init__(self, value=value, n=n)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.propose_n = propose_n
        self.propose_step = propose_step
        self.prior = self.compute_prior()

    def propose(self):
        """Propose a new GrammarHypothesis; used to propose new samples with methods like MH.

        num_values => propose_n
        step_size => propose_step

        New value is sampled from a normal centered @ old values, w/ proposal as covariance (inverse?)

        TODO: should we randomly select our rules? or is there a better way

        """
        step = self.propose_step * np.random.multivariate_normal([0.]*self.n, self.proposal)
        newv = self.value

        # change `propose_n` number of values/rules
        for i in random.sample(range(len(newv)), self.propose_n):
            if newv[i] + step[i] > 0.0:
                newv[i] += step[i]

        c = self.__copy__()
        c.set_value(newv)
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

        This is taken as a weighted sum over all hypotheses.

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
            posteriors = [h.compute_posterior(d.input)[0] for h in hypotheses]
            Z = logsumexp(posteriors)

            for o in d.output.keys():
                # probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                p = logsumexp([(post-Z) if (not h() is None) and (o in h()) else -Infinity
                               for h, post in zip(hypotheses,posteriors)])
                k = d.output[o][0]         # num. yes responses
                n = k + d.output[o][1]     # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        self.likelihood = likelihood
        self.update_posterior()
        return likelihood

    def rule_distribution(self, data, rule_name, vals=np.arange(0, 2, .1)):
        """Compute posterior values for this grammar, varying specified rule over a set of values."""
        rule_index = self.get_rule_index(rule_name)
        return self.conditional_distribution(data, rule_index, vals=vals)

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

    def get_rule(self, rule_name):
        """Get the GrammarRule associated with this rule name."""
        rule_index = self.get_rule_index(rule_name)
        return self.get_rule_by_index(rule_index)

    def get_rules(self, rule_name):
        """Get all GrammarRules associated with this rule name."""
        return [r for r in self.rules if r.name == rule_name]

    def get_rule_by_index(self, rule_index):
        """Get the GrammarRule at this index."""
        return self.rules[rule_index]

    def get_rule_index(self, rule_name):
        """Get index of the GrammarRule associated with this rule name."""
        rule_indexes = [i for i, r in enumerate(self.rules) if r.name == rule_name]
        assert (len(rule_indexes) == 1), "ERROR: More than 1 rule associated with this rule name!!!"
        return rule_indexes[0]

    def __copy__(self):
        """Make a shallow copy of this GrammarHypothesis."""
        return GrammarHypothesis(
            self.grammar, self.hypotheses,
            value=copy.copy(self.value), n=self.n, proposal=copy.copy(self.proposal),
            prior_shape=self.prior_shape, prior_scale=self.prior_scale,
            propose_n=self.propose_n, propose_step=self.propose_step
        )
