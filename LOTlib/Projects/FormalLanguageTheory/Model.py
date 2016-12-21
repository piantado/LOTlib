
from copy import deepcopy, copy
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import IDR_proposal
from LOTlib.Eval import TooBigException
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException

MAX_STRINGS = 2000 # if we generate more than this, we trim the computation

class InnerHypothesis(LOTHypothesis):
    """s
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda C, lex_, x: %s", **kwargs): # lexicon, x arg, context
        LOTHypothesis.__init__(self, grammar=grammar, display=display,  recurse_bound=10, **kwargs)


    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = IDR_proposal(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

    def __call__(self, *args):
        return flatten2str(LOTHypothesis.__call__(self, *args), sep="" )


class MyException(Exception):
    """ Used to escape recursion here"""
    pass

from LOTlib.Flip import *

from LOTlib.Miscellaneous import flatten2str
from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLogPrefixDistance
from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
class IncrementalLexiconHypothesis( MultinomialLikelihoodLogPrefixDistance, SimpleLexicon):
        """ A hypothesis where we can incrementally add words """

        def __init__(self, grammar=None, **kwargs):
            SimpleLexicon.__init__(self,  maxnodes=50, **kwargs)
            self.grammar=grammar # the base gramar (with 0 included); we copy and add in other recursions on self.deepen()
            self.N = 0 # the number of meanings we have
            self.outlier = -1000.0 # read in MultinomialLikelihood
            self.max_total_calls = 30 # this is the most internal recurse_ calls we can do without raising an exception
            self.total_calls = 0
            self.distance = 10.0 # penalize

        def make_hypothesis(self, **kwargs):
            return InnerHypothesis(**kwargs) # no default grammar since it will differ by word

        def propose(self):

            new = deepcopy(self)  ## Now we just copy the whole thing
            while True:
                try:
                    # i = sample_one(range(self.N)) # random one
                    i = max(self.value.keys()) # only propose to last
                    x, fb = self.get_word(i).propose()
                    new.set_word(i, x)

                    new.grammar = self.value[0].grammar # keep the grammar the same object

                    return new, fb

                except ProposalFailedException:
                    pass

        def dispatch_word(self, context, word, x):
            """ We override this so that the hypothesis defaultly calls with the last word via __call__"""

            # keep track of too many calls
            self.total_calls += 1
            if self.total_calls > self.max_total_calls:
                raise MyException

            # call this word
            return self.value[word](context, self.dispatch_word, x)  # pass in "self" as lex, using the recursive version

        def recursive_call(self, word, *args):
            raise NotImplementedError

        def deepen(self):
            """ Add one more word here. This requires making a new grammar and setting the word to recurse to the
             previous one so that all of our hard work is not lost."""

            # update the grammar
            grammar = deepcopy(self.grammar)

            for n in xrange(1,self.N+1): # add in all the rules up till the next one
                grammar.add_rule('SELFF', '%s' % (n,), None, 1.0)  # add the new N # needed for the new grammar, wdd what we will use

            initfn                 = grammar.get_rule_by_name('', nt='START').make_FunctionNodeStub(grammar, None)
            initfn.args[0]         = grammar.get_rule_by_name("lex_(C, %s, %s)").make_FunctionNodeStub(grammar, initfn)
            initfn.args[0].args[0] = grammar.get_rule_by_name('%s' % (self.N-1)).make_FunctionNodeStub(grammar, initfn.args[0])
            initfn.args[0].args[1] = grammar.get_rule_by_name('x').make_FunctionNodeStub(grammar, initfn.args[0])

            self.set_word(self.N, self.make_hypothesis(value=initfn, grammar=grammar)) # set N to what it should, using our new grammar

            self.N += 1

        def reset_and_call(self, *args):
            """ Call returns a dictinoary. This calls once, reseting the total_calls counter.
                We can't pass dispatch_word since that doesn't reset the total_call counter
            """
            self.total_calls = 0
            return self.dispatch_word(*args)

        def __call__(self):
            """
            Following the MultinomialLikelihood we want a dictionary fromm outcomes to LPs
            """
            assert self.N > 0, "*** Cannot call IncrementalLexiconHypothesis unless N>0"

            try:
                return compute_outcomes(self.reset_and_call, self.N-1, '',  alsocatch=(MyException, TooBigException))
            except TooManyContextsException:
                return {'':0.0} # return nothing
