
from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis


class JoshConceptsHypothesis(LOTHypothesis):
    """Class for josh concepts number game with independent rule probs."""
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        Probability is alpha if datum in set, (1-alpha) otherwise).

        """
        s = self()      # set of numbers corresponding to this hypothesis
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(datum):
            if s is not None and datum in s:
                return log(self.alpha/len(s) + error_p)
            else:
                return log(error_p)

        likelihoods = [compute_single_likelihood(d) for d in data]
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.update_posterior()
        return self.likelihood


class MixtureGrammarHypothesis(GrammarHypothesis):
    """
    This will let us single out 'MATH' rules & 'INTERVAL' rules as `lambda` & `(1-lambda)`.

    """
    def __init__(self, grammar, hypothesis, **kwargs):
        GrammarHypothesis.__init__(grammar, hypothesis, propose_n=1, **kwargs)

    def set_value(self, value):
        """
        This is what's different for this class. We only have one value -- lambda in our mixture model.

        We use lembda then to re-set our grammar rules (hand code: MATH == lambda, INTERVAL == (1-lambda).)

        """
        assert isinstance(value, list) and len(value) == 1, "ERROR: Invalid value!!!"
        self.value = value
        lambda_mix = value[0]

        # Set probability for each rule corresponding to value index
        for rule in self.get_rules(rule_to='MATH'):
            rule.p = lambda_mix
        for rule in self.get_rules(rule_to='INTERVAL'):
            rule.p = 1 - lambda_mix

        # Recompute hypothesis priors, given new grammar probs
        for h in self.hypotheses:
            self.grammar.recompute_generation_probabilities(h.value)
            h.compute_prior()

