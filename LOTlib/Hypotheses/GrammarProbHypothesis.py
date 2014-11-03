
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

"""
import numpy as np
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity


class GrammarProbHypothesis(VectorHypothesis):
    """Hypothesis for grammar so we can represent rule prob. assignments as a vector.

    Inherits from VectorHypothesis, though I haven't figured out yet whe this really means...

    Attributes:
        grammar (LOTlib.Grammar): The grammar.
        hypotheses (LOTlib.Hypothesis): List of hypotheses, generated beforehand.
        rules (list): List of all rules in the grammar.
        value (list): Vector of numbers corresponding to the items in `rules`.

        TODO:
            - should this have a 'proposal' arg? (for VectorHypothesis)
            - compute_prior - is this right? what about shape/scale parameters?

    """
    def __init__(self, grammar, hypotheses, **kwargs):
        self.rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        p_vector = [rule.p for rule in self.rules]
        n = len(p_vector)
        VectorHypothesis.__init__(self, value=p_vector, n=n, proposal=np.eye(n))
        self.hypotheses = hypotheses
        self.grammar = grammar
        self.prior = self.compute_prior()

    def compute_prior(self):
        # TODO what should shape & scale values here be?
        shape = 1.0
        return np.random.gamma(shape, scale=1.0, size=self.n)

    def compute_likelihood(self, data):
        """Use hypotheses to estimate likelihood of generating the data.

        This is taken as a weighted sum over all hypotheses.

        Args:
            input_data(list): List of input integers.
            output_data(dict):

        Returns:
            dict: Each output key returns the summed likelihood of that single data point. Keys are the
            same as those of argument `output_data`.

        """
        hypotheses = self.hypotheses
        Z = logsumexp([h.posterior_score for h in hypotheses])
        likelihood = -Infinity

        for h in hypotheses:
            w = h.posterior_score - Z
            old_likelihood = h.likelihood
            # TODO: is h.compute_likelihood updating posterior_score each loop?
            weighted_likelihood = h.compute_likelihood([data]) + w
            h.likelihood = old_likelihood
            likelihood = logplusexp(likelihood, weighted_likelihood)

        self.posterior_score = self.prior + self.likelihood
        return likelihood

    def prob_output_datum(self, output_int, output_datum):
        """Compute the probability of generating human data given our grammar & input data.

        Args:
            output_data (tuple): (# yes, # no) responses in [human] evaluation data.

        Returns:
             float: Estimated probability of generating human data.

        """
        p = self.compute_likelihood(output_int)
        k = output_datum[0]         # num. yes responses
        n = k + output_datum[1]     # num. trials
        bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient  # TODO is this right?
        return bc + (k*p) + (n-k)*log1mexp(p)                   # log version
        # p_data = bc * pow(p, k) * pow(1-p, n-k)               # linear version
        # bc = factorial(n) / (factorial(k) * factorial(n-k))

    def prob_output_data(self, output_data):
        """
        Input is a dict with keys ~ output ints, e.g. '8' or '16', and values are # yes & no responses.

        """
        return logsumexp([self.prob_output_datum(d.key(), d.item()) for d in output_data])

    def dist_over_rule(self, data, rule_name, probs=np.arange(0, 2, .2)):
        r_index = self.get_rule_index(rule_name)
        dist = []
        old_value = self.value.copy()
        for p in probs:
            value = self.value.copy()
            value[r_index] = p
            self.set_value(value)
            dist.append(self.prob_output_data(data))

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
            h.log_probability()   # TODO figure out how to do this

    def get_rule(self, rule_index):
        """Get the GrammarRule at this index."""
        return self.rules[rule_index]

    def get_rule_index(self, rule_name):
        """Get rules index associated with this rule name."""
        # there should only be 1 item in this list
        rules = [i for i, r in enumerate(self.rules) if r.name == rule_name]
        return rules[0]



