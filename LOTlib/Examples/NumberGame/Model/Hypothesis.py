from math import log
import numpy
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logplusexp
# from LOTlib.Evaluation.Eval import *
import Inference as I, Grammar as G


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Domain-specific hypothesis wrapper class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class NumberSetHypothesis(LOTHypothesis):
    """Wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a set of numbers.
    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_likelihood(self, data):
        """Likelihood of specified data being produced by this hypothesis. If datum item not in set,
        it still has (1 - alpha) likelihood of being generated.
        """
        h = self.__call__()         # get set of numbers corresponding to this hypothesis
        alpha = self.alpha
        noise = (1-alpha) / self.domain
        likelihood = 0
        for datum in data:
            if datum in h:
                likelihood += log(alpha/len(h) + noise)
            else:
                likelihood += log(noise)

        self.likelihood = likelihood
        self.posterior_score = self.prior + likelihood      # required in all compute_likelihoods
        return likelihood


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Grammar hypothesis class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class GrammarProbHypothesis(VectorHypothesis):
    """Hypothesis for grammar so we can represent rule prob. assignments as a vector.

    Inherits from VectorHypothesis, though I haven't figured out yet whe this really means...

    Attributes:
        grammar (LOTlib.Grammar): The grammar.
        rules (list): List of all rules in the grammar.
        value (list): Vector of numbers corresponding to the items in `rules`.
        alpha (float): Noise parameter.
        domain (int): Number corresponding to the max of our integer domain. E.g. domain=100 ~ range[1,100]
        p_table (dict): Table of likelihood values stored for computed likelihoods. Keys in this correspond
            to probability vectors, and values correspond to the weighted likelihoods calculated for these.

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        p_vector = [rule.p for rule in rules]
        n = len(p_vector)
        VectorHypothesis.__init__(self, value=p_vector, n=n, proposal=numpy.eye(n))

        self.rules = [rule for sublist in G.grammar.rules.values() for rule in sublist]
        self.grammar = grammar
        self.alpha = alpha
        self.domain = domain

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
            p_rule = I.prob_data_rule(G.grammar, rule, data, p, num_iters, alpha)
            p_grammar = logplusexp(p_grammar, p_rule)

        return p_grammar
