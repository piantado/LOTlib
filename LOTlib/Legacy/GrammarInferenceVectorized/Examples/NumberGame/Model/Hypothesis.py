
from math import log
from LOTlib.FunctionNode import FunctionNode, BVUseFunctionNode
from LOTlib.Evaluation.EvaluationException import TooBigException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis, Infinity
from LOTlib.Miscellaneous import attrmem


class NumberGameHypothesis(LOTHypothesis):
    """
    Domain-specific wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a subset of integers in [1, domain].

    """
    def __init__(self, grammar, value=None, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, value=value, args=[], **kwargs)
        self.grammar = grammar
        self.alpha = alpha
        self.domain = domain
        self.value_set = None

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability."""
        # Re-compute the FunctionNode `self.value` generation probabilities
        self.grammar.log_probability(self.value)

        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            prior = -Infinity
        else:
            # Compute prior with either RR or not.
            prior = self.grammar.log_probability(self.value) / self.prior_temperature

        # Don't use this tree if we have 2 constants as children in some subnode OR 2 BV's
        for fn in self.value.subnodes()[1:]:
            args = [i for i in fn.argFunctionNodes()]
            # TODO: 0 prior for double OPCONST wasn't working - it assigned 0 prior to other things, e.g. [y1 ends-in 5]
            # if all([arg.name == '' and len(arg.args)>1 and isinstance(arg.args[0], FunctionNode)
            #         and arg.args[0].returntype=='OPCONST' for arg in fn.argFunctionNodes()]) \
            #         or (all([isinstance(arg, BVUseFunctionNode) for arg in fn.argFunctionNodes()]) and len(args) > 1):

            if (all([(arg.returntype=='OPCONST') for arg in args])                                                      # reject if all OPCONST children
                    or all([isinstance(arg, BVUseFunctionNode) for arg in fn.argFunctionNodes()])) and len(args) > 1:   # OR if 2 BV children
                prior = -Infinity
                break

        if len(self()) == 0:
            prior = -Infinity

        self.update_posterior()
        return prior

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        Args:
            data (FunctionData): this is the data; we only use data.input
            update_post (bool): boolean -- do we update posterior?

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

        likelihoods = [compute_single_likelihood(d) for d in data.input]
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

            if isinstance(value_set, set):
                # Restrict our concept to being within our domain
                value_set = [x for x in value_set if (1 <= x <= self.domain)]
            else:
                # Sometimes self() returns None
                value_set = set()
            self.value_set = value_set

        return self.value_set

    def compile_function(self):
        self.value_set = None
        return LOTHypothesis.compile_function(self)

    def __copy__(self, value=None):
        return NumberGameHypothesis(self.grammar, value=value, alpha=self.alpha, domain=self.domain)


