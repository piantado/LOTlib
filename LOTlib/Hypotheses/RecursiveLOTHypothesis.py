
from LOTHypothesis import LOTHypothesis, raise_exception
from LOTlib.Evaluation.EvaluationException import RecursionDepthException, TooBigException, EvaluationException

class RecursiveLOTHypothesis(LOTHypothesis):
    """
    A LOTHypothesis that permits recursive calls to itself via the primitive "recurse" (previously, L).

    Here, RecursiveLOTHypothesis.__call__ does essentially the same thing as LOTHypothesis.__call__, but it binds
    the symbol "recurse" to RecursiveLOTHypothesis.recursive_call so that recursion is processed internally.

    This bind is done in compile_function, NOT in __call__

    For a Demo, see LOTlib.Examples.Number
    """

    def __init__(self, grammar, recurse='recurse_', recurse_bound=25, args=['x'], **kwargs):
        """
        Initializer. recurse gives the name for the recursion operation internally.
        """

        # save recurse symbol
        self.recurse = recurse
        self.recursive_depth_bound = recurse_bound # how deep can we recurse?
        self.recursive_call_depth = 0 # how far down have we recursed?

        # automatically put 'recurse' onto args
        assert args[0] is not recurse # not already specified
        args = [recurse] + args

        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)

    def recursive_call(self, *args):
        """
        This gets called internally on recursive calls. It keeps track of the depth and throws an error if you go too deep
        """

        self.recursive_call_depth += 1

        if self.recursive_call_depth > self.recursive_depth_bound:
            raise RecursionDepthException

        # Call with sending myself as the recursive call
        return LOTHypothesis.__call__(self, self.recursive_call, *args)

    def __call__(self, *args):
        """
        The main calling function. Resets recursive_call_depth and then calls
        """
        self.recursive_call_depth = 0

        # call with passing self.recursive_Call as the recursive call
        return LOTHypothesis.__call__(self, self.recursive_call, *args)

