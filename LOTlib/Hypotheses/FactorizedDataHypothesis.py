"""
An example of a "factorized data" model where instead of having a function F generate the data,
we have a family of functions, each of which generates part of the data from the previous parts.
"""

import random
from copy import deepcopy

from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Proposers.RegenerationProposer import RegenerationProposer
from LOTlib.Hypotheses.Proposers.InsertDeleteProposer import InsertDeleteProposer
from LOTlib.Evaluation.EvaluationException import TooBigException


class InnerHypothesis(StochasticFunctionLikelihood, RecursiveLOTHypothesis, RegenerationProposer, InsertDeleteProposer):
    """
    The type of each function F.
    """
    def __init__(self, grammar=None, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:
            return ''

    def propose(self):
        if random.random() < 0.5:
            return RegenerationProposer.propose(self)
        else:
            return InsertDeleteProposer.propose(self)


class FactorizedDataHypothesis(SimpleLexicon):
    """
        An abstract class where we write the data as a composition of functions.

        self.__call__ calls using a compositional structure (which we may want to change in the future) of
        the ith function takes all the previous i outputs as arguments, and we return the last one.

        A SimpleLexicon associating each integer n with an InnerHypothesis. Each InnerHypothesis' grammar
        must be augmented with the arguments for the previous f_i

        This requires self.make_hypothesis to be defined and take a grammar argument.
    """
    def __init__(self, N=4, grammar=None, argument_type='LIST', variable_weight=2.0, value=None, **kwargs):

        SimpleLexicon.__init__(self, value=value)

        self.N = N

        if grammar is not None: # else we are in a copy initializer, and the rest will get copied
            for w in xrange(N):
                nthgrammar = deepcopy(grammar)

                # Add all the bound variables
                args = [  ]
                for xi in xrange(w):  # no first argument
                    argi = 'x%s'%xi

                    # Add a rule for the variable
                    nthgrammar.add_rule(argument_type, argi, None, variable_weight)

                    args.append(argi)

                # and add a rule for the n-ary recursion
                nthgrammar.add_rule('LIST', 'recurse_', [argument_type]*(w), 1.)

                self.set_word(w, self.make_hypothesis(grammar=nthgrammar, args=args))

    def __call__(self):
        # The call here must take no arguments. If this changes, alter x%si above
        theargs = []
        v = ''
        for w in xrange(self.N):
            try:
                v = self.get_word(w)(*theargs) # call with all prior args
                theargs.append(v)
            except TooBigException:
                theargs.append('')
            # print "V=", v, theargs

        return v # return the last one

    def make_hypothesis(self, **kwargs):
        raise NotImplementedError


class FactorizedLambdaHypothesis(SimpleLexicon):
    """
        A modified version of FactorizedDataHypothesis, where we pass the lambda function of previous InnerHypothesis
        as the parameter to next InnerHypothesis instead of the value of it.

        We made it with two tricks: We construct and pass the lambda expression to next InnerHypothesis; We wrap the
        expression inside the recurse_() function with a lambda function to make it callable for next level of recursion
    """
    def __init__(self, N=4, grammar=None, argument_type='LIST', variable_weight=2.0, value=None, **kwargs):

        SimpleLexicon.__init__(self, value=value)

        self.N = N
        self.call_time = 0

        if grammar is not None: # else we are in a copy initializer, and the rest will get copied
            for w in xrange(N):
                nthgrammar = deepcopy(grammar)

                # Add all the bound variables
                args = [  ]
                for xi in xrange(w):  # no first argument
                    argi = 'x%s'%xi

                    # Add a rule for the variable
                    nthgrammar.add_rule(argument_type, argi, [''], variable_weight)

                    args.append(argi)

                # and add a rule for the n-ary recursion
                nthgrammar.add_rule('LIST', 'recurse_', ['FUNCTION']*(w), 1.)
                # we wrap the content with lambda to make it callable for next recursion level
                nthgrammar.add_rule('FUNCTION', 'lambda', ['LIST'], 1.)

                self.set_word(w, self.make_hypothesis(grammar=nthgrammar, args=args))

    def __call__(self):
        # The call here must take no arguments. If this changes, alter x%si above
        args_list = [[]]
        v = lambda: ''
        for w in xrange(self.N):
            # pass the callable version of this hypothesis to next one
            v = lambda: self.try_run(self.get_word, args_list)
            if w != self.N-1: args_list.append(args_list[w]+[v])
            # print "V=", v, theargs

        return v() # return the last one

    def make_hypothesis(self, **kwargs):
        raise NotImplementedError

    def try_run(self, f, arg):
        ind = self.get_ind()
        self.call_time += 1
        try:
            return f(ind)(*(arg[ind]))
        except TooBigException:
            return ''
        except RuntimeError:
            return ''

    def get_ind(self):
        return self.N - 1 - self.call_time % self.N