
from copy import deepcopy, copy
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import IDR_proposal
from LOTlib.Eval import TooBigException
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException

MAX_STRINGS = 2000 # if we generate more than this, we trim the computation

class InnerHypothesis(RecursiveLOTHypothesis):
    """s
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda recurse_, lex_: %s", **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, display=display,  recurse_bound=10, **kwargs)


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

    def recursive_call(self, *args):
        """
        This was overwritten to NOT raise an exception, but rather return an emptry stirng. That way we can still generate all the shorter strings.
        An exception (Default behavior) prevents all other reursive calls from executing
        """

        self.recursive_call_depth += 1

        if self.recursive_call_depth > self.recursive_depth_bound:
            return {'': 0.0}

        v = LOTHypothesis.__call__(self, self.recursive_call, *args)

        # and prevent us from dealing with gigantic sets of strings
        if len(v.keys()) > MAX_STRINGS:
            raise MyException

        return v

class MyException(Exception):
    pass



from collections import Counter
from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLogPrefixDistance
from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
class IncrementalLexiconHypothesis( MultinomialLikelihoodLogPrefixDistance, SimpleLexicon):
        """ A hypothesis where we can incrementally add words and propose to only the additions
        """

        def __init__(self, grammar=None, **kwargs):
            SimpleLexicon.__init__(self,  maxnodes=50, variable_weight=3.0, **kwargs)
            self.grammar=grammar
            self.N = 0
            self.outlier = -100 # read in MultinomialLikelihood
            self.max_total_calls = 100 # this is the most internal recurse_ calls we can do without raising an exception

        def make_hypothesis(self, **kwargs):
            return InnerHypothesis(**kwargs)

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

        def dispatch_word(self, word):
            """ We override this so that the hypothesis defaultly calls with the last word via __call__"""

            if self.total_calls > self.max_total_calls: raise MyException

            # handle the 0th word which may call lex inappropriately
            if word < 0:
                raise MyException

            if word in self.fmem:
                return self.fmem[word]
            else:
                # it's funny, hre we don't pass mem since the args to InnerHypothesis don't need the arguments
                # to recurse. They only need to know recurse's name, and the grammar takes care of passing it
                # the memoized argument above. There are two versions, one for recursing to myself and one
                # for recursing to another word. If I go to another word, I reset the recursion depth (since that
                # word must be earlier)

                v = self.value[word](self.dispatch_word)  # pass in "self" as lex, using the recursive version

                self.total_calls += self.value[word].recursive_call_depth # how many recurses did we do?

                # things that can give excpetions: to big, or too much recursing
                if len(v.keys()) > MAX_STRINGS: raise MyException

                self.fmem[word] = v
                return v

        def recursive_call(self, word, *args):
            raise NotImplementedError

        def deepen(self):
            """ Add one more word here """

            # update the grammar
            self.grammar = deepcopy(self.grammar)

            # update the hypothesis.
            if self.N == 0: # if we're empty
                initfn = self.grammar.generate()
            else: # else automatically recurse to the current word in order to avoid losing thresete likelihood
                self.grammar.add_rule('SELFF', '%s' % (self.N-1), None, 1.0)

                initfn                 = self.grammar.get_rule_by_name('', nt='START').make_FunctionNodeStub(self.grammar, None)
                initfn.args[0]         = self.grammar.get_rule_by_name("lex_(%s)").make_FunctionNodeStub(self.grammar, initfn)
                initfn.args[0].args[0] = self.grammar.get_rule_by_name('%s' % (self.N-1)).make_FunctionNodeStub(self.grammar, initfn.args[0])


            self.set_word(self.N, self.make_hypothesis(value=initfn, grammar=self.grammar))

            self.N += 1

        def __call__(self, *args):
            """
            Wrap in self as a first argument that we don't have to in the grammar. This way, we can use self(word, X Y) as above.
            """
            self.total_calls = 0
            self.fmem = dict() # zero out

            assert len(args) == 0, "*** Currently no args are supported"

            try:
                return self.dispatch_word(self.N - 1) # defaultly, a call to h() is a call to the last word and it won't be memoized, so h() samples forward
            except (MyException, TooBigException):
                return {'': 0.0}
