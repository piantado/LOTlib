from LOTlib.Evaluation.Eval import * # Necessary for compile_function eval below
from LOTlib.Evaluation.EvaluationException import TooBigException, EvaluationException
from LOTlib.Hypotheses.FunctionHypothesis import FunctionHypothesis
from LOTlib.Hypotheses.Proposers.RegenerationProposer import RegenerationProposer
from LOTlib.Miscellaneous import Infinity, raise_exception, attrmem

class LOTHypothesis(FunctionHypothesis, RegenerationProposer):
    """A FunctionHypothesis built from a grammar.

    Arguments
    ---------
    grammar : LOTLib.Grammar
        The grammar for the hypothesis.
    value : FunctionNode
        The value for the hypothesis.
    maxnodes : int
        The maximum amount of nodes that the grammar can have
    args : list
        The arguments to the function.

    Attributes
    ----------
    grammar_vector : np.ndarray
        This is a vector of
    prior_vector : np.ndarray

    """

    def __init__(self, grammar=None, value=None, f=None, start=None, ALPHA=0.9, maxnodes=25, args=['x'], **kwargs):

        # Save all of our keywords
        self.__dict__.update(locals())

        if value is None and grammar is not None:
            value = grammar.generate()

        FunctionHypothesis.__init__(self, value=value, f=f, args=args, **kwargs)

        self.likelihood = 0.0
        self.rules_vector = None

    def __call__(self, *args):
        # NOTE: This no longer catches all exceptions.
        try:
            return FunctionHypothesis.__call__(self, *args)
        except TypeError as e:
            print "TypeError in function call: ", e, str(self), "  ;  ", type(self), args
            raise TypeError
        except NameError as e:
            print "NameError in function call: ", e, " ; ", str(self), args
            raise NameError

    def type(self):
        return self.value.type()

    def compile_function(self):
        """Called in set_value to compile into a function."""
        if self.value.count_nodes() > self.maxnodes:
            return lambda *args: raise_exception(TooBigException)
        else:
            try:
                return eval(str(self)) # evaluate_expression(str(self))
            except Exception as e:
                print "# Warning: failed to execute evaluate_expression on " + str(self)
                print "# ", e
                return lambda *args: raise_exception(EvaluationException)

    def compute_single_likelihood(self, datum):
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------------------
    # Compute prior

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability.

        """
        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            return -Infinity
        else:
            # Compute prior with either RR or not.
            return self.grammar.log_probability(self.value) / self.prior_temperature
