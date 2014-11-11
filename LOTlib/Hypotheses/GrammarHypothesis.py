
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

"""
import copy
import numpy as np
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
    # TODO should this have a 'proposal' arg? (for VectorHypothesis)
    def __init__(self, grammar, hypotheses, **kwargs):
        self.rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        self.grammar = grammar
        self.hypotheses = hypotheses
        p_vector = [rule.p for rule in self.rules]
        n = len(p_vector)
        VectorHypothesis.__init__(self, value=p_vector, n=n, proposal=np.eye(n))
        self.prior = self.compute_prior()

    def compute_prior(self):
        # TODO what should shape & scale values here be?
        shape = 1.0
        rule_priors = np.random.gamma(shape, scale=1.0, size=self.n)
        return logsumexp(rule_priors)

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
        Z = logsumexp([h.posterior_score for h in hypotheses])
        likelihood = -Infinity

        for d in data:
            for h in hypotheses:
                h.compute_posterior(d.input)
                w = h.posterior_score - Z
                h_copy = copy.copy(h)

                for key in d.output.keys():
                    # calculate Pr (output_data==Yes | h)
                    # TODO: is h.compute_likelihood updating posterior_score each loop?
                    p = h.compute_likelihood([key]) + w

                    # calculate Pr (data of this hypothesis == human data)
                    k = d.output[key][0]         # num. yes responses
                    n = k + d.output[key][1]     # num. trials
                    bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient  # TODO is this right?
                    human_likelihood = bc + (k*p) + (n-k)*log1mexp(p)       # likelihood that we get human output
                    # p_data = bc * pow(p, k) * pow(1-p, n-k)               ## linear version
                    # bc = factorial(n) / (factorial(k) * factorial(n-k))
                    likelihood = logplusexp(likelihood, human_likelihood)

                h = h_copy
        self.likelihood = likelihood
        self.posterior_score = self.prior + self.likelihood

    def rule_distribution(self, data, rule_name, probs=np.arange(0, 2, .2)):
        """Compute posterior values for this grammar, varying specified rule over a set of probabilites.

        Args:
            data(list): List of FunctionData objects.
            rule_name(string): Name of the rule we're varying probabilities over.  E.g. 'union_'
            probs(list): List of float probability values.  E.g. [0,.2,.4, ..., 2.]

        Returns:
            list: List of posterior scores, where each item corresponds to an item in `probs` argument.

        """
        r_index = self.get_rule_index(rule_name)
        dist = []
        old_value = copy.copy(self.value)
        for p in probs:
            value = copy.copy(self.value)
            value[r_index] = p
            self.set_value(value)
            dist.append(self.compute_likelihood(data))

        self.set_value(old_value)
        return dist

    def set_value(self, value):
        """Set value and grammar rules for this hypothesis."""
        if not (len(value) == len(self.rules)):
            print "INVALID VALUE VECTOR"
            return
        self.value = value
        for i in range(1, len(value)):
            self.rules[i].p = value[i]
        for h in self.hypotheses:
            h.compute_prior()   # TODO figure out how to do this

    def get_rule(self, rule_index):
        """Get the GrammarRule at this index."""
        return self.rules[rule_index]

    def get_rule_index(self, rule_name):
        """Get rules index associated with this rule name."""
        # there should only be 1 item in this list
        rules = [i for i, r in enumerate(self.rules) if r.name == rule_name]
        return rules[0]



