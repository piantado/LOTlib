
from math import log
import numpy as np
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis, Infinity


# ------------------------------------------------------------------------------------------------------------
# NumberGameHypothesis simplified to resemble 'Josh Concepts' model

class JoshConceptsHypothesis(LOTHypothesis):
    """Domain hypothesis class for josh concepts number game with independent rule probs."""
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


# ------------------------------------------------------------------------------------------------------------
# GrammarHypothesis classes

class MixtureGrammarHypothesis(GrammarHypothesis):
    """
    This will let us single out 'MATH' rules & 'INTERVAL' rules as `lambda` & `(1-lambda)`.

    """
    def __init__(self, grammar, hypotheses, value=None, **kwargs):
        if value is None:
            value = np.array([1.0, 1.0])
        GrammarHypothesis.__init__(self, grammar, hypotheses, value=value, **kwargs)

    def set_value(self, value):
        """
        This is what's different for this class. We only have one value -- lambda in our mixture model.

        We use lambda then to re-set our grammar rules (hand code: MATH == lambda, INTERVAL == (1-lambda).)

        """
        assert len(value) == 2, "ERROR: Invalid value!!!"
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.n = len(value)

        # TODO: possibly also just keep 1 of these fixed at 1...
        # Set probability for each rule corresponding to value index
        for rule in self.get_rules(rule_to='MATH')[1]:
            rule.p = value[0]
        for rule in self.get_rules(rule_to='INTERVAL')[1]:
            rule.p = value[1]

        self.update()

    def get_propose_idxs(self):
        """Get indexes to propose to => for Mix...Hypothesis, we use a special self.value system."""
        return [0, 1]


class ShallowGrammarHypothesis(GrammarHypothesis):
    """
    Here, we only do values that come from the 'START' node (i.e. 2-depth trees)

    """
    def __init__(self, grammar, hypotheses, value=None, **kwargs):
        if not value:
            value = [r.p for r in self.get_rules(rule_nt='START')]
        GrammarHypothesis.__init__(self, grammar, hypotheses, value=value, **kwargs)

    def set_value(self, value):
        assert len(value) is self.n, "ERROR: Invalid value!!!"
        self.value = value
        propose_rules = [r for r in self.get_rules(rule_nt='START')]
        for i in range(self.n):
            propose_rules[i] = value[i]

        self.update()
