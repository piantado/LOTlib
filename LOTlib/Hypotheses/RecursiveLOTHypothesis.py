
from LOTHypothesis import LOTHypothesis, raise_exception, evaluate_expression
from LOTlib.Evaluation.EvaluationException import RecursionDepthException, TooBigException, EvaluationException

class RecursiveLOTHypothesis(LOTHypothesis):
    """
    A LOTHypothesis that permits recursive calls to itself via the primitive "recurse" (previously, L).

    Here, RecursiveLOTHypothesis.__call__ does essentially the same thing as LOTHypothesis.__call__, but it binds
    the symbol "recurse" to RecursiveLOTHypothesis.recursive_call so that recursion is processed internally.

    For a Demo, see LOTlib.Examples.Number

    NOTE: Pre Nov2014, this was computed with some fanciness in evaluate_expression that automatically appended the Y combinator.
          This change was made to simplify and speed things up.
    """

    def __init__(self, grammar, recurse='recurse_', recurse_bound=25, **kwargs):
        """
        Initializer. recurse gives the name for the recursion operation internally.
        """

        # save recurse symbol
        self.recurse = recurse
        self.recursive_depth_bound = recurse_bound # how deep can we recurse?
        self.recursive_call_depth = 0 # how far down have we recursed?

        # automatically put 'recurse' onto kwargs['args']
        assert recurse not in kwargs['args'] # not already specified
        kwargs['args'] = [recurse] + kwargs['args']

        LOTHypothesis.__init__(self, grammar, **kwargs)

    def recursive_call(self, *args):
        """
        This gets called internally on recursive calls. It keeps track of the depth to allow us to escape
        """
        self.recursive_call_depth += 1
        if self.recursive_call_depth > self.recursive_depth_bound:
            raise RecursionDepthException

        return LOTHypothesis.__call__(self, *args)

    def __call__(self, *args):
        """
        The main calling function. Resets recursive_call_depth and then calls
        """
        self.recursive_call_depth = 0
        return LOTHypothesis.__call__(self, *args)


    def compile_function(self):
        """
        Called in set_value to make a function. Here, we defaultly wrap in recursive_call as our argument to "recurse"
        so that recursive_call gets called by the symbol self.recurse
        """
        """Called in set_value to compile into a function."""
        if self.value.count_nodes() > self.maxnodes:
            return (lambda *args: raise_exception(TooBigException))
        else:
            try:
                # Here, we evaluate it, and then defaultly pass recursive_call as the first "Recurse"
                f = evaluate_expression(str(self))
                return lambda *args: f(self.recursive_call, *args)
            except Exception as e:
                print "# Warning: failed to execute evaluate_expression on " + str(self)
                print "# ", e
                return (lambda *args: raise_exception(EvaluationException) )