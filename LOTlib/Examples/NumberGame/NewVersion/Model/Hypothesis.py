
from math import log
from LOTlib.Evaluation.EvaluationException import TooBigException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis


class NumberGameHypothesis(LOTHypothesis):
    """Domain-specific wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a subset of integers in [1, domain].

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain
        self.value_set = None

    def compute_single_likelihood(self, datum, s, error_p, updateflag=True):
        if s is not None and datum in s:
            return log(self.alpha/len(s) + error_p)
        else:
            return log(error_p)

    def compute_likelihood(self, data, updateflag=True, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        """
        try:
            s = self()      # set of numbers corresponding to this hypothesis
                            # NOTE: This may be None if the hypothesis has too many nodes
        except OverflowError:
            s = set()       # If our hypothesis call blows things up...   TODO handle this better!!
        error_p = (1.-self.alpha) / self.domain

        likelihoods = [self.compute_single_likelihood(d, s, error_p, updateflag=updateflag) for d in data]
        likelihood = sum(likelihoods) / self.likelihood_temperature
        if updateflag:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    def __call__(self, *args, **kwargs):
        if self.value_set is None:
            try:                    # Sometimes self.value has too many nodes!
                value_set = LOTHypothesis.__call__(self)
            except TooBigException:
                value_set = set()
            # Restrict our concept to being within our domain
            if isinstance(value_set, set):
                value_set = [x for x in value_set if x <= self.domain]
            else:
                value_set = set()   # Sometimes self() returns None
            self.value_set = value_set

        return self.value_set

    def compile_function(self):
        self.value_set = None
        return LOTHypothesis.compile_function(self)

    def __copy__(self, copy_value=False):
        return NumberGameHypothesis(self.grammar, alpha=self.alpha, domain=self.domain)


