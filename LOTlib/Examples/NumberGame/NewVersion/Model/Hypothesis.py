
from math import log
from LOTlib.Evaluation.EvaluationException import TooBigException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis, Infinity


class NumberGameHypothesis(LOTHypothesis):
    """
    Domain-specific wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a subset of integers in [1, domain].

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain
        self.value_set = None

    def compute_prior(self, recompute=False, vectorized=False):
        """Compute the log of the prior probability.

        """
        # Re-compute the FunctionNode `self.value` generation probabilities
        if recompute:
            self.value.recompute_generation_probabilities(self.grammar)

        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            self.prior = -Infinity
        else:
            # Compute prior with either RR or not.
            self.prior = self.value.log_probability() / self.prior_temperature

        # Don't use this tree if we have 2 constants as children in some subnode
        for fn in self.value.subnodes()[1:]:
            if all([arg.name == '' and len(arg.args)==1 and isinstance(arg.args[0], FunctionNode)
                                   and arg.args[0].returntype=='OPCONST' for arg in fn.argFunctionNodes()]):
                self.prior = -Infinity
                break

        self.update_posterior()
        return self.prior

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        """
        try:
            s = self()      # Set of numbers corresponding to this hypothesis
        except OverflowError:
            s = set()       # If our hypothesis call blows things up
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(d):
            """Internal method so we don't have to call self() each time."""
            if s is not None and d in s:
                return log(self.alpha/len(s) + error_p)
            else:
                return log(error_p)

        likelihoods = [compute_single_likelihood(d) for d in data]
        likelihood = sum(likelihoods) / self.likelihood_temperature
        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    def compute_single_likelihood(self, d):
        try:
            s = self()
        except OverflowError:
            s = set()
        error_p = (1.-self.alpha) / self.domain

        if s is not None and d in s:
            return log(self.alpha/len(s) + error_p)
        else:
            return log(error_p)

    def __call__(self, *args, **kwargs):
        if self.value_set is None:
            # Sometimes self.value has too many nodes
            try:
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


