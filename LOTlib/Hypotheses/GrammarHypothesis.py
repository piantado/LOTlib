
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

"""
import copy
import numpy as np
from scipy.stats import gamma
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity, log


class GrammarHypothesis(VectorHypothesis):
    """Hypothesis for grammar, we can represent grammar rule probability assignments as a vector.

    Inherits from VectorHypothesis, though I haven't figured out yet whe this really means...

    Attributes:
        grammar (LOTlib.Grammar): The grammar.
        hypotheses (LOTlib.Hypothesis): List of hypotheses, generated beforehand.
        rules (list): List of all rules in the grammar.
        value (list): Vector of numbers corresponding to the items in `rules`.

    """
    def __init__(self, grammar, hypotheses, prior_shape=2., prior_scale=1., **kwargs):
        self.rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        self.grammar = grammar
        self.hypotheses = hypotheses
        p_vector = [rule.p for rule in self.rules]
        n = len(p_vector)
        VectorHypothesis.__init__(self, value=p_vector, n=n)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.prior = self.compute_prior()

    def compute_prior(self):
        shape = self.prior_shape
        scale = self.prior_scale
        rule_priors = [gamma.logpdf(r, shape, scale=scale) for r in self.value]

        prior = sum([r for r in rule_priors])      # TODO is this right?
        self.prior = prior
        self.posterior_score = self.prior + self.likelihood
        return prior

    def compute_likelihood(self, data, **kwargs):
        """Use hypotheses to estimate likelihood of generating the data.

        This is taken as a weighted sum over all hypotheses.

        Args:
            data(list): List of FunctionData objects.

        Returns:
            float: Likelihood summed over all outputs, summed over all hypotheses & weighted for each
            hypothesis by posterior score p(h|d).

        """
        hypotheses = self.hypotheses
        likelihood = -Infinity

        for d in data:
            for h in hypotheses: h.compute_posterior(d.input)
            Z = logsumexp([h.posterior_score for h in hypotheses])

            for h in hypotheses:
                w = h.posterior_score - Z

                for key in d.output.keys():
                    # calculate Pr (output_data==Yes | h)
                    p = h.compute_likelihood([key])

                    # calculate Pr (data of this hypothesis == human data)
                    k = d.output[key][0]         # num. yes responses
                    n = k + d.output[key][1]     # num. trials
                    bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                    likelihood_human_data = bc + (k*p) + (n-k)*log1mexp(p)  # likelihood that we get human output
                    likelihood = logplusexp(likelihood, likelihood_human_data + w)

        self.likelihood = likelihood
        self.posterior_score = self.prior + self.likelihood
        return likelihood

    def rule_distribution(self, data, rule_name, vals=np.arange(0, 2, .2)):
        """Compute posterior values for this grammar, varying specified rule over a set of values."""
        rule_index = self.get_rule_index(rule_name)
        return self.conditional_distribution(data, rule_index, vals=vals)

    def set_value(self, value):
        """Set value and grammar rules for this hypothesis."""
        if not (len(value) == len(self.rules)):
            print '%'*80+"INVALID VALUE VECTOR\n"+'%'*80
            return
        self.value = value
        for i in range(1, len(value)):
            self.rules[i].p = value[i]
        for h in self.hypotheses:
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
        rules = [i for i, r in enumerate(self.rules) if r.name == rule_name]
        assert (len(rules) == 1), 'ERROR: More than 1 rule associated with this rule name!'
        return rules[0]
