"""
An example of a "factorized data" model where instead of having a function F generate the data,
we have a family of functions, each of which generates part of the data from the previous parts.
"""

import random
from copy import deepcopy

from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Proposers.Regeneration import regeneration_proposal
from LOTlib.Hypotheses.Proposers.InsertDelete import insert_delete_proposal
from LOTlib.Eval import TooBigException


class InnerHypothesis(StochasticLikelihood, RecursiveLOTHypothesis):
    """
    The type of each function F.
    """
    def __init__(self, grammar=None, display="lambda recurse_: %s", **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, display=display, **kwargs)

    # def __call__(self, *args):
    #     try:
    #         return RecursiveLOTHypothesis.__call__(self, *args)
    #     except RecursionDepthException:
    #         return ''

    def propose(self):
        if random.random() < 0.5:
            ret = regeneration_proposal(self.grammar, self.value)

        else:
            ret = insert_delete_proposal(self.grammar, self.value)

        p = Hypothesis.__copy__(self, value=ret[0])
        ret[0] = p
        return ret

#
# class FactorizedDataHypothesis(SimpleLexicon):
#     """
#         An abstract class where we write the data as a composition of functions.
#
#         self.__call__ calls using a compositional structure (which we may want to change in the future) of
#         the ith function takes all the previous i outputs as arguments, and we return the last one.
#
#         A SimpleLexicon associating each integer n with an InnerHypothesis. Each InnerHypothesis' grammar
#         must be augmented with the arguments for the previous f_i
#
#         This requires self.make_hypothesis to be defined and take a grammar argument.
#     """
#     def __init__(self, N=4, grammar=None, argument_type='LIST', variable_weight=2.0, value=None, **kwargs):
#
#         SimpleLexicon.__init__(self, value=value)
#
#         self.N = N
#
#         if grammar is not None: # else we are in a copy initializer, and the rest will get copied
#             for w in xrange(N):
#                 nthgrammar = deepcopy(grammar)
#
#                 # Add all the bound variables
#                 args = [  ]
#                 for xi in xrange(w):  # no first argument
#                     argi = 'x%s'%xi
#
#                     # Add a rule for the variable
#                     nthgrammar.add_rule(argument_type, argi, None, variable_weight)
#
#                     args.append(argi)
#
#                 # and add a rule for the n-ary recursion
#                 nthgrammar.add_rule('LIST', 'recurse_', [argument_type]*(w), 1.)
#
#                 self.set_word(w, self.make_hypothesis(grammar=nthgrammar, args=args))
#
#     def __call__(self):
#         # The call here must take no arguments. If this changes, alter x%si above
#         theargs = []
#         v = ''
#         for w in xrange(self.N):
#             try:
#                 v = self.get_word(w)(*theargs) # call with all prior args
#                 theargs.append(v)
#             except TooBigException:
#                 theargs.append('')
#             # print "V=", v, theargs
#
#         return v # return the last one
#
#     def make_hypothesis(self, **kwargs):
#         raise NotImplementedError


class FactorizedLambdaHypothesis(SimpleLexicon):
    """
        A modified version of FactorizedDataHypothesis, where we pass the lambda function of previous InnerHypothesis
        as the parameter to next InnerHypothesis instead of the value of it.

        We made it with two tricks: We construct and pass the lambda expression to next InnerHypothesis; We wrap the
        expression inside the recurse_() function with a lambda function to make it callable for next level of recursion
    """
    def __init__(self, N=4, grammar=None, argument_type='FUNCTION', variable_weight=2.0, value=None, recurse_bound=25, **kwargs):

        SimpleLexicon.__init__(self, value=value)
        self.base_grammar = deepcopy(grammar)
        self.argument_type = argument_type
        self.variable_weight = variable_weight
        self.recurse_bound = recurse_bound

        self.recursive_call_depth = 0

        if grammar is not None: # else we are in a copy initializer, and the rest will get copied
            self.N = 0

            for w in xrange(N):
                self.add_new_word()

    def add_new_word(self):
        """ add the k'th word to the model"""

        nthgrammar = deepcopy(self.base_grammar)

        # Add all the bound variables
        args = []
        for xi in xrange(self.N):  # no first argument
            argi = 'x%s' % xi

            # Add a rule for the variable
            nthgrammar.add_rule(self.argument_type, argi, None, self.variable_weight)

            args.append(argi)

        # and add a rule for the n-ary recursion
        nthgrammar.add_rule('LIST', 'recurse_', ['FUNCTION'] * (self.N), 1.)
        # we wrap the content with lambda to make it callable for next recursion level
        nthgrammar.add_rule('FUNCTION', 'lambda', ['LIST'], 1.)
        nthgrammar.add_rule('LIST', '(%s)()', ['FUNCTION'], 1.)

        arg_str = "lambda recurse_"
        for argi in args:
            arg_str = arg_str + ", " + argi
        arg_str = arg_str + ": %s"

        self.set_word(self.N, self.make_hypothesis(grammar=nthgrammar, display=arg_str))

        self.N += 1 # we have one more word

    def __call__(self):
        # The call here must take no arguments. If this changes, alter x%si above
        theargs = []
        v = lambda: ''
        for w in xrange(self.N):
            # pass the callable version of this hypothesis to next one
            f = self.get_word(w); arg = deepcopy(theargs)

            v = lambda f=f, arg=arg: self.try_run(f, arg)
            theargs.append(v)
            # print "V=", v, theargs

        self.recursive_call_depth = 0
        try:
            return v() # return the last one
        except (RecursionDepthException, TooBigException):
            return ''

    def make_hypothesis(self, **kwargs):
        raise NotImplementedError

    def try_run(self, f, arg):
        # This just calls f on args, but raises the recursion (or TooBigException) that we need

        self.recursive_call_depth += 1

        if self.recursive_call_depth  > self.recurse_bound:
            raise RecursionDepthException

        return f(*arg)
