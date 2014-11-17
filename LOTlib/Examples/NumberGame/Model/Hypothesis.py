
from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity
# from LOTlib.Evaluation.Eval import *


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Domain-specific hypothesis wrapper class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class NumberGameHypothesis(LOTHypothesis):
    """Wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a set of numbers.

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_single_likelihood(self, datum):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        """
        s = self()     # set of numbers corresponding to this hypothesis
                       # NOTE: This may be None if the hypothesis has too many nodes
        error_p = (1.-self.alpha) / self.domain

        if s is not None and datum in s:
            likelihood = log(self.alpha/len(s) + error_p)
        else:
            likelihood = log(error_p)
        return likelihood

    def compute_likelihood(self, data, **kwargs):
        """Sum likelihoods over all data points, divide by likelihood_temperature."""
        likelihoods = map(self.compute_single_likelihood, data)
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.update_posterior()
        return self.likelihood








'''
#### OLD STUFF

    def compute_likelihood(self, data, num_iters=10000, alpha=0.9):
        """Compute the likelihood of producing the human data given the input `data` & this `grammar`.

        The way this is currently computed is that we st

        Args:
            data (list): This is a list of (input data, human data) pairs; see Inference.probs_data_rule
            for an example of the data format.

        Returns:
            float: likelihood of this grammar relative to `data` (in log space?)

        """
        p_grammar = 0
        for i in range(self.n):
            rule = self.rules[i]
            p = self.value[i]
            p_rule = I.prob_data_rule(self.grammar, rule, data, p, num_iters, alpha)
            p_grammar = logplusexp(p_grammar, p_rule)

        return p_grammar




    # in GrammarProbHypothesis...
    def set_value(self, value):
        """ Set value and grammar rules for this hypothesis"""
        self.value = value
        # now update the hypotheses to correspond to the new values entered...
        rules = self.rules
        for i in range(0, len(value)):
            p = value
            rules[i].p = value[i]
        self.hypotheses = self.h_sample(self.grammar)

    def sample_hypotheses(self, input_data, num_iters=10000, alpha=0.9):
        """Generate new hypotheses by sampling w/ metro hastings."""
        hypotheses = set()
        h0 = U.make_h0(self.grammar, alpha=alpha)
        for h in MHSampler(h0, input_data, steps=num_iters):
            hypotheses.add(h)

        return hypotheses



'''






