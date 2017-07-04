import sys
from copy import copy, deepcopy
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import IDR_proposal
from LOTlib.Eval import TooBigException
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursionDepthException
from LOTlib.Miscellaneous import flatten2str, sample_one, attrmem
from LOTlib.Flip import *
from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import *
from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.Primitives.Strings import StringLengthException
from LOTlib.Hypotheses.Proposers.InsertDeleteRegenerationProposer import InsertDeleteRegenerationProposer

from Levenshtein import editops # for distance likelihood only

MAX_SELF_RECURSION = 100 # how many times can a hypothesis call itself? NOTE: This bounds the length of strings, as in a^n
sys.setrecursionlimit(10000) # needed for generating long strings

class InnerHypothesis(InsertDeleteRegenerationProposer, LOTHypothesis):
    """s
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda C, lex_, x: %s", **kwargs): # lexicon, x arg, context
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

#from cachetools import lru_cache

# @lru_cache(10000)
def edit_likelihood(x,y, alphabet_size=2, noise=0.01):

    ops = editops(x,y)
    lp = log(1.0-noise)*len(y) # all the unchanged
    for o, _, _ in ops:
        if   o == 'equal':   lp += log(noise) - log(4.0)
        elif o == 'replace': lp += log(noise) - log(4.0) - log(alphabet_size)
        elif o == 'insert':  lp += log(noise) - log(4.0) - log(alphabet_size)
        elif o == 'delete':  lp += log(noise) - log(4.0)
        else: assert False
    # print lp, x, y
    return lp

class IncrementalLexiconHypothesis(SimpleLexicon):
        """ A hypothesis where we can incrementally add words """

        def __init__(self, grammar=None, alphabet_size=2, **kwargs):
            SimpleLexicon.__init__(self,  **kwargs)
            self.grammar=grammar # the base gramar (with 0 included); we copy and add in other recursions on self.deepen()
            self.N = 0 # the number of meanings we have
            self.max_total_calls = MAX_SELF_RECURSION # this is the most internal recurse_ calls we can do without raising an exception It gets increased every deepen(). NOTE: if this is small, then it bounds the length of each string
            self.total_calls = 0
            self.alphabet_size = alphabet_size

        def make_hypothesis(self, **kwargs):
            return InnerHypothesis(**kwargs) # no default grammar since it will differ by word

        @attrmem('prior')
        def compute_prior(self):
            return SimpleLexicon.compute_prior(self) - self.N * log(2.0)/self.prior_temperature # coin flip for each additional word

        def compute_single_likelihood(self, datum):
            assert isinstance(datum.output, dict)

            hp = self(*datum.input)  # output dictionary, output->probabilities
            assert isinstance(hp, dict)

            s = 0.0
            for k, dc in datum.output.items():
                if k in hp:
                    s += dc * hp[k]
                elif len(hp.keys()) > 0:
                    s += dc * min([ edit_likelihood(x, k, alphabet_size=self.alphabet_size) for x in hp.keys() ]) # the highest probability string; or we could logsumexp
                else:
                    s += dc * edit_likelihood('', k, alphabet_size=self.alphabet_size)

                # This is the mixing {a,b}* noise model
                # lp = log(1.0-datum.alpha) - log(self.alphabet_size+1)*(len(k)+1) #the +1s here count the character marking the end of the string
                # if k in hp:
                #     lp = logplusexp(lp, log(datum.alpha) + hp[k]) # if non-noise possible
                # s += dc*lp
            return s

        def propose(self):

            new = copy(self)  ## Now we just copy the whole thing

            while True:
                try:
                    i = sample_one(range(self.N)) # random one
                    # i = max(self.value.keys()) # only propose to last
                    x, fb = self.get_word(i).propose()
                    new.set_word(i, x)

                    return new, fb

                except ProposalFailedException:
                    pass

        def deepen(self):
            """ Add one more word here. This requires making a new grammar and setting the word to recurse to the
             previous one so that all of our hard work is not lost."""

            # update the grammar
            grammar = deepcopy(self.grammar)

            for n in xrange(1,self.N+1): # add in all the rules up till the next one
                grammar.add_rule('SELFF', '%s' % (n,), None, 1.0)  # add the new N # needed for the new grammar, wdd what we will use

            initfn                         = grammar.get_rule_by_name('', nt='START').make_FunctionNodeStub(grammar, None)
            initfn.args[0]                 = grammar.get_rule_by_name('', nt='START2').make_FunctionNodeStub(grammar, initfn)
            initfn.args[0].args[0]         = grammar.get_rule_by_name("lex_(C, %s, %s)").make_FunctionNodeStub(grammar, initfn.args[0])
            initfn.args[0].args[0].args[0] = grammar.get_rule_by_name('%s' % (self.N-1), nt='SELFF').make_FunctionNodeStub(grammar, initfn.args[0].args[0])
            initfn.args[0].args[0].args[1] = grammar.get_rule_by_name('x').make_FunctionNodeStub(grammar, initfn.args[0].args[0])

            self.set_word(self.N, self.make_hypothesis(value=initfn, grammar=grammar)) # set N to what it should, using our new grammar

            self.max_total_calls += MAX_SELF_RECURSION
            self.N += 1

        def dispatch_word(self, context, word, x):
            """ We override this so that the hypothesis defaultly calls with the last word via __call__"""

            # keep track of too many calls
            self.total_calls += 1
            if self.total_calls > self.max_total_calls:
                raise RecursionDepthException

            # call this word
            v = self.value[word](context, self.dispatch_word, x)  # pass in "self" as lex, using the recursive version

            return v

        def recursive_call(self, word, *args):
            raise NotImplementedError

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
            # print ">>>>>>", self
            try:
                return compute_outcomes(self.reset_and_call, self.N-1, '',  maxcontext=200, maxit=200, catchandpass=(RecursionDepthException, StringLengthException))
            except (TooManyContextsException, TooBigException):
                return {'':0.0} # return nothing
