
from copy import deepcopy, copy
from math import log
from LOTlib.Miscellaneous import attrmem, logsumexp, sample_one
# from Levenshtein import distance
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import IDR_proposal
from LOTlib.Eval import TooBigException

class InnerHypothesis(LOTHypothesis):
    """
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda recurse_: %s", **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display=display, **kwargs)


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

class MyException(Exception):
    pass

from collections import Counter
from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLogPrefixDistance
from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
class IncrementalLexiconHypothesis( MultinomialLikelihoodLogPrefixDistance, RecursiveLexicon):
        """ A hypothesis where we can incrementally add words and propose to only the additions
        """

        def __init__(self, grammar=None, **kwargs):
            RecursiveLexicon.__init__(self, recursive_depth_bound=50, maxnodes=50, variable_weight=3.0, **kwargs)
            self.grammar=grammar
            self.N = 0
            self.outlier = -100 # read in MultinomialLikelihood

        def make_hypothesis(self, **kwargs):
            return InnerHypothesis(**kwargs)

        def propose(self):

            new = deepcopy(self)  ## Now we just copy the whole thing
            while True:
                try:
                    i = sample_one(range(self.N)) # random one
                    # i = max(self.value.keys()) # only propose to last
                    x, fb = self.get_word(i).propose()
                    new.set_word(i, x)

                    new.grammar = self.value[0].grammar # keep the grammar the same object

                    return new, fb

                except ProposalFailedException:
                    pass

        def dispatch_word(self, word, memoized):
            """ We override this so that the hypothesis defaultly calls with the last word via __call__"""
            self.recursive_call_depth += 1
            # print self.recursive_call_depth, str(self)

            if self.recursive_call_depth > self.recursive_depth_bound:
                return {'':0.0} # empty string, log p
            else:

                if memoized and word in self.fmem:
                    return self.fmem[word]
                else:
                    # it's funny, hre we don't pass mem since the args to InnerHypothesis don't need the arguments
                    # to recurse. They only need to know recurse's name, and the grammar takes care of passing it
                    # the memoized argumetn above
                    v = self.value[word](self.dispatch_word)  # pass in "self" as lex, using the recursive version

                    if len(v.keys()) > 2000: raise MyException

                    self.fmem[word] = v
                    return v

        def recursive_call(self, word, *args):
            raise NotImplementedError

        def deepen(self):
            """ Add one more word here """

            # update the grammar
            self.grammar = deepcopy(self.grammar)
            self.grammar.add_rule('SELFF', '%s' % (self.N), None, 1.0)

            # update the hypothesis.
            if self.N == 0: # if we're empty
                initfn = self.grammar.generate()
            else: # else automatically recurse to the current word in order to avoid losing the likelihood
                initfn                 = self.grammar.get_rule_by_name('', nt='START').make_FunctionNodeStub(self.grammar, None)
                initfn.args[0]         = self.grammar.get_rule_by_name("recurse_(%s, False)").make_FunctionNodeStub(self.grammar, initfn)
                initfn.args[0].args[0] = self.grammar.get_rule_by_name('%s' % (self.N)).make_FunctionNodeStub(self.grammar, initfn.args[0])

            self.set_word(self.N, self.make_hypothesis(value=initfn, grammar=self.grammar))

            self.N += 1

        def __call__(self, *args):
            """
            Wrap in self as a first argument that we don't have to in the grammar. This way, we can use self(word, X Y) as above.
            """
            self.recursive_call_depth = 0
            self.fmem = dict() # zero out

            assert len(args) == 0, "*** Currently no args are supported"

            try:
                return self.dispatch_word(self.N - 1, False) # defaultly, a call to h() is a call to the last word and it won't be memoized, so h() samples forward
            except MyException:
                return {'': 0.0}
