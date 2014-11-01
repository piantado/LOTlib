"""

TODO:
    - make generic grammar hypothesis class, number game version should be an extension of that
    - add temperature to NumberGameHypothesis
    - should compute_likelihood use logsumexp?

"""
from math import log
import numpy
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logplusexp, logsumexp
# from LOTlib.Evaluation.Eval import *
import Utilities as U, Grammar as G


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
        h = self.__call__()         # get set of numbers corresponding to this hypothesis
        alpha = self.alpha
        noise = (1-alpha) / self.domain

        if datum in h:
            likelihood = log(alpha/len(h) + noise)
        else:
            likelihood = log(noise)
        return likelihood

    def compute_likelihood(self, data):
        """Sum likelihoods over all data points, divide by likelihood_temperature."""
        likelihoods = map(self.compute_single_likelihood, data)
        self.likelihood = logsumexp(likelihoods) / self.likelihood_temperature
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood



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

        h_sample: Hypothesis sampling function. This is a (lambda grammar: function) that should take a
        grammar as input (input data for MH should be pre-parameterized..?), and return a sampled set of
        hypotheses . . .

        E.g.
        >> data = [1,2,3]
        >> mh_sample = lambda grammar: set(MHSampler(make_h0(grammar), data, steps=10000))
        >> h = GrammarProbHypothesis(grammar, mh_sample)

        TODO:
            - should this have a 'proposal' arg? (for VectorHypothesis)
            - compute_prior - is this right? what about shape/scale parameters?
            - prob_output_datum - logify factorials

    """
    def __init__(self, grammar, h_sample, alpha=0.9, domain=100, **kwargs):
        self.rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        p_vector = [rule.p for rule in self.rules]
        n = len(p_vector)
        VectorHypothesis.__init__(self, value=p_vector, n=n, proposal=numpy.eye(n))
        self.hypotheses = h_sample(grammar)
        self.grammar = grammar
        self.alpha = alpha
        self.domain = domain
        self.prior = self.compute_prior()

    def compute_prior(self):
        shape = 1.0
        return numpy.random.gamma(shape, scale=1.0, size=self.n)

    def compute_likelihood(self, data, num_iters=10000, alpha=0.9):
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
        Z = U.normalizing_constant(hypotheses)
        likelihood = -U.Infinity

        for h in hypotheses:
            w = h.posterior_score - Z
            old_likelihood = h.likelihood
            # TODO: is h.compute_likelihood updating posterior_score each loop?
            weighted_likelihood = h.compute_likelihood([data]) + w
            h.likelihood = old_likelihood
            likelihood = logplusexp(likelihood, weighted_likelihood)

        self.posterior_score = self.prior + self.likelihood
        return likelihood

    def prob_output_datum(self, output_datum):
        """Compute the probability of generating human data given our grammar & input data.

        Args:
            grammar (LOTlib.Grammar): The grammar.
            output_data (list): Tuple corresponding to (# yes, # no) responses in [human] evaluation data.

        Returns:
             float: Estimated probability of generating human data.

        TODO:
            log-ify factorials

        """
        p = self.compute_likelihood(output_datum)
        k = output_datum[0]         # num. yes responses
        n = k + output_datum[1]     # num. trials
        bc = factorial(n) / (factorial(k) * factorial(n-k))     # binomial coefficient
        return log(bc) + (k*p) + (n-k)*log1mexp(p)              # log version
        # p_data = bc * pow(p, k) * pow(1-p, n-k)               # linear version

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
#### OLDOLDOLD

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







'''








### generic grammar prob class OMG
### ==> all this stuff is old, currently

class GrammarHypothesis(VectorHypothesis):
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

