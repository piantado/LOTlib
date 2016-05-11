"""
    Trying a version that's more like a pipeline

    x = F(x)


"""

from LOTlib.Eval import register_primitive, RecursionDepthException
@register_primitive
def bY(fn, recurse_count, bound=25):
    # A bounded Y-combinator.
    # takes a function lambda x,slf: ... and return a lambda x: .. where slf is bound
    # to itself, but the max recursions allowed is bounded
    def inner_bounded_f(x):
        recurse_count[0] += 1
        if recurse_count[0] > bound or len(x) > 100:
            raise RecursionDepthException
        else:
            return fn(x, inner_bounded_f)

    return inner_bounded_f


from LOTlib.Grammar import Grammar

grammar = Grammar(start='PROGRAM')

grammar.add_rule('PROGRAM', '%s',         ['STATEMENT'], 1.0 )
grammar.add_rule('PROGRAM', '%s\n    %s', ['STATEMENT', 'PROGRAM'], 1.0 ) # hmm the order here may matter?

grammar.add_rule('STATEMENT', 'x = x+(bY(%s, recurse_count))(x)', ['FUNCTION'], 1.0)

grammar.add_rule('FUNCTION', 'lambda <BV>, slf: %s', ['REXPRESSION'], 1.0, bv_type='VARIABLE') # just give it a default value (in case of no lambdas)


grammar.add_rule('EXPRESSION', 'x', None, 1.0)
grammar.add_rule('EXPRESSION', '(%s+%s)', ['EXPRESSION', 'EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s[0])', ['EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s[1:])', ['EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s if %s else %s)', ['EXPRESSION', 'BOOL', 'EXPRESSION'], 1.0)
grammar.add_rule('BOOL', 'empty_', ['EXPRESSION'], 1.)
grammar.add_rule('BOOL', 'flip_', [''], 1.)


grammar.add_rule('REXPRESSION', 'slf(%s)', ['REXPRESSION'], 1.0)
grammar.add_rule('REXPRESSION', 'x', None, 1.0)
grammar.add_rule('REXPRESSION', '(%s+%s)', ['REXPRESSION', 'REXPRESSION'], 1.0)
grammar.add_rule('REXPRESSION', '(%s[0])', ['REXPRESSION'], 1.0)
grammar.add_rule('REXPRESSION', '(%s[1:])', ['REXPRESSION'], 1.0)
grammar.add_rule('REXPRESSION', '(%s if %s else %s)', ['REXPRESSION', 'RBOOL', 'REXPRESSION'], 1.0)
grammar.add_rule('RBOOL', 'empty_', ['REXPRESSION'], 1.)
grammar.add_rule('RBOOL', 'flip_', [''], 1.)

from copy import deepcopy
eng_grammar = deepcopy(grammar)
# TODO: Be sure the probs line up with EXPRESSION
for t in 'abc':
    grammar.add_rule('EXPRESSION', "'%s'"%t, None, 1.0) ## Define expressions
    grammar.add_rule('REXPRESSION', "'%s'"%t, None, 1.0) # These are the expressions allows in the rhs of functions -- same as above, but slf is permitted

for t in 'davtn':
    eng_grammar.add_rule('EXPRESSION', "'%s'"%t, None, 1.0) ## Define expressions
    eng_grammar.add_rule('REXPRESSION', "'%s'"%t, None, 1.0) # These are the expressions allows in the rhs of functions -- same as above, but slf is permitted

# Make a string for "compiling" hypotheses
# Note this includes a default initialization for y, and allowes G to call itself it if wants
DISPLAY = """
def G(x=''):
    recurse_count = [0]
    %s
    return x
"""
# Or to a BV, above

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Eval import EvaluationException

from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log
from random import random

from LOTlib.FunctionNode import BVAddFunctionNode
def last_bv_string(x, nt, bvt, d=0):
    """ Give me the last/deepest bv name introduced on an nt with bv type bvt
        NOTE: This is dependent on the string mapping in pystring
    """

    below = [ last_bv_string(n,nt, bvt, d=d+1) for n in x.argFunctionNodes() ]

    below = [ v for v in below if v is not None] # remove the Nones

    if len(below) > 0:
        return below[-1] # return the last
    else: # nothing below
        if isinstance(x, BVAddFunctionNode) and x.returntype == nt and x.added_rule.nt == bvt:
            return x.added_rule.bv_prefix+str(d)
        else:
            return None

from LOTlib.Hypotheses.Proposers import regeneration_proposal, insert_delete_proposal, ProposalFailedException

def prefix_distance(x,y):
    m = max(len(x), len(y))
    cnt = 0
    for i in xrange(min(len(x), len(y))):
        if x[i] == y[i]:
            cnt += 1
        else:
            break
    return m - cnt

class LanguageHypothesis(StochasticFunctionLikelihood,LOTHypothesis):
    def __init__(self, **kwargs):
        """
            fixed_ll_counts: should be set to a value in evaluating
        """
        LOTHypothesis.__init__(self, max_nodes=200, display=DISPLAY, **kwargs)
        self.fixed_ll_counts = None


    def propose(self, **kwargs):

        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ## TODO NOTE: The fb probs should take into account the probability of the proposal type

                if random() < 0.5:
                    ret_value, fb = regeneration_proposal(self.grammar, self.value) # don't unpack, since we may return [newt,fb] or [newt,f,b]
                else:
                    ret_value, fb = insert_delete_proposal(self.grammar, self.value) # don't unpack, since we may return [newt,fb] or [newt,f,b]

                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

    def compile_function(self):
        # Override the default compilation to define g

        exec str(self) in locals() # define a local function called G

        return G # G takes a "recurse" argument that is automatically included through RecursiveLOTHypothesis

    def make_ll_counts(self, input, nsamples=512):
        """
            We'll catch the errors here and return empty counter, so that we don't eval many times with errors
        """

        if self.fixed_ll_counts is not None:
            return self.fixed_ll_counts

        try:
            return StochasticFunctionLikelihood.make_ll_counts(self, input, nsamples=nsamples)
        except (IndexError, RuntimeError, EvaluationException, RecursionDepthException): # catch y3=y3 kinds of errors, or ''[0], or infinite recursion -- ugh!
            self.ll_counts =  {'':nsamples} # If any error, give back nothing
            return self.ll_counts


    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input)

        lo = sum(llcounts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 100.0 * distance(r,k) for r in llcounts.keys() ])
            # ll += datum.output[k] * min([ log(llcounts[r])-log(lo) - 100.0 * distance(r,k) for r in llcounts.keys() ])
        return ll

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
fff = sys.stdout.flush


def make_new_hypothesis(s, **kwargs):
    """
        NOTE: grammar only has atom a, you need to add other atoms yourself
    """
    mk_grammar = eng_grammar if s == 'SimpleEnglish' else grammar
    return LanguageHypothesis(grammar=mk_grammar, **kwargs)

if __name__=="__main__":


    from LOTlib import break_ctrlc
    from LOTlib.DataAndObjects import FunctionData

    # data = [FunctionData(input=[], output={'a':10, 'aa':5, 'aaa':2, 'aaaa':1})]
    # data = [FunctionData(input=[], output={'ab':20, 'aabb':10, 'aaabbb':5, 'aaaabbbb':2, 'aaaaabbbbb':1})]
    data = [FunctionData(input=[], output={'abc':64, 'aabbcc':32, 'aaabbbccc':16, 'aaaabbbbcccc':8, 'aaaaabbbbbccccc':4, 'aaaaaabbbbbbcccccc':2})]

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    h0 = LanguageHypothesis(prior_temperature=1.0, likelihood_temperature=1.0, grammar=grammar)
    for h in break_ctrlc(MHSampler(h0, data, skip=10)):
        print h.posterior_score, h.prior, h.likelihood
        print h
        print len(h.value)
        print "LL COUNTS ARE:", h.ll_counts
        print "------------"
        pass