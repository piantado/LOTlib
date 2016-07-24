from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData
import random
from LOTlib.Miscellaneous import attrmem, Infinity, nicelog

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(size=100):
    #onset: h coda: N unrestricted: f s n m
    return [FunctionData(input=[],
                         #output={'k i s': 100, 'm i m': 100, 'n a s': 100, 'm a k': 100, 'h i s': 100, 'm i f': 100, 's a f': 100, 'n a N': 100, 'm a s': 100, 'k i m': 100, 'g a k': 100, 'm i s': 100, 's i f': 100, 'h a n': 100, 'n i n': 100, 's a n': 100, 'f i f': 100, 's i n': 100, 'm a n': 100, 'n a m': 100, 'f a m': 100, 'n i f': 100, 'k i N': 100, 'g a s': 100, 'f a n': 100, 'f a s': 100, 'm a f': 100, 'g a N': 100, 'h a s': 100, 'n a f': 100, 'n i s': 100, 'k i n': 100, 'f a k': 100, 'n i g': 100, 'k i g': 100, 'h i g': 100, 'g a m': 100, 'k i f': 100, 'h a k': 100, 'h i N': 100, 's a N': 100, 'f i m': 100, 'h a m': 100, 'm a N': 100, 's i N': 100, 's i g': 100, 'n a n': 100, 's a m': 100, 'h i f': 100, 'm i g': 100, 's i s': 100, 'm i n': 100, 'h i n': 100, 'm i N': 100, 'f i n': 100, 'h a N': 100, 's i m': 100, 's a k': 100, 'f i g': 100, 'n a k': 100, 'g a n': 100, 'h a f': 100, 'm a m': 100, 'g a f': 100, 'f i s': 100, 'n i N': 100})]
                         output={'n i k': 100, 'h i N': 100, 'f a n': 100, 'g i f': 100, 'm a N': 100, 'n a s': 100, 'g i k': 100, 'k a n': 100, 'n i f': 100, 'f a g': 100, 'g i m': 100, 'g i s': 100, 's i f': 100, 'k i m': 100, 'n i m': 100, 'g a s': 100, 'k a f': 100, 'f a s': 100, 's i n': 100, 's a f': 100, 's i k': 100, 's i m': 100, 'h i m': 100, 'h i n': 100, 'f a N': 100, 'h i k': 100, 'k a m': 100, 'h i f': 100, 'f a m': 100, 'g i N': 100, 'm i f': 100, 'n i s': 100, 'k i N': 100, 's i N': 100, 'n a m': 100, 'h i s': 100, 'f i s': 100, 'k a s': 100, 'g a n': 100, 'g a m': 100, 'h a f': 100, 'k i s': 100, 'm i n': 100, 'k a N': 100, 'g a f': 100, 'g i n': 100, 'k a g': 100, 's a n': 100, 's a m': 100, 'n a f': 100, 'n a g': 100, 'm i N': 100, 's a g': 100, 'f i k': 100, 'h a N': 100, 'f i n': 100, 'f i m': 100, 'm a s': 100, 'g a N': 100, 'h a s': 100, 'k i f': 100, 'n a N': 100, 'm i s': 100, 's a N': 100, 'm i k': 100, 'h a g': 100, 'm a g': 100, 'm a f': 100, 'k i n': 100, 'h a m': 100, 'h a n': 100, 'n i N': 100, 'f i N': 100, 'm a n': 100})]


# if vowel is a: g is onset, k is coda. if vowel is i: g is coda k is onset.
#h is always onset. N is always coda
#unrestricted sounds: f s n m

import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

@primitive
def vowels_():
    return "aeiou"

TERMINAL_WEIGHT = 20

grammar = Grammar()

grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'sample_', ['SET'], 1.0)
grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0/1.5)

grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)
grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)

grammar.add_rule('EXPR', 'if_', ['BOOL', 'EXPR', 'EXPR'], 1./4)
#grammar.add_rule('BOOL', 'equal_', ['EXPR', 'EXPR'], 1./10)
grammar.add_rule('BOOL', 'flip_', [''], 1.)

#grammar.add_rule('SET', 'vowels_()', None, 1.0)
grammar.add_rule('TERMINAL', 'g', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'a', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'i', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'k', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 's', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'f', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'n', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'm', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'h', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'N', None, TERMINAL_WEIGHT)

## Allow lambda abstraction
'''grammar.add_rule('EXPR', 'apply_', ['LAMBDAARG', 'LAMBDATHUNK'], 1./10)
grammar.add_rule('LAMBDAARG',   'lambda', ['EXPR'], 1./10, bv_type='EXPR', bv_args=[] )
grammar.add_rule('LAMBDATHUNK', 'lambda', ['EXPR'], 1./10, bv_type=None, bv_args=None ) # A thunk'''


from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
#from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy

from LOTlib.Miscellaneous import logsumexp
#from Levenshtein import distance
from math import log

class MyHypothesis(StochasticLikelihood, LOTHypothesis):
#class MyHypothesis(StochasticLevenshteinLikelihood, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda : %s', **kwargs)


    #overwrite propose
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = numpy.random.choice([insert_delete_proposal,regeneration_proposal])(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=-Infinity, nsamples=512, sm=0.1, **kwargs):
        # For each input, if we don't see its input (via llcounts), recompute it through simulation

        ll = 0.0
        for datum in data:
            self.ll_counts = self.make_ll_counts(datum.input, nsamples=nsamples)
            z = sum(self.ll_counts.values())
            ll += sum([datum.output[k]*(nicelog(self.ll_counts[k]+sm) - nicelog(z+sm*len(datum.output.keys()))) for k in datum.output.keys()])
            if ll < shortcut:
                return -Infinity

        return ll / self.likelihood_temperature

    #overwrite compute_single_likelihood to alter distance factor
    '''def compute_single_likelihood(self, datum, distance_factor=1000.0):
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"
        llcounts = self.make_ll_counts(datum.input)
        lo = sum(llcounts.values()) # normalizing constant
        # We are going to compute a pseudo-likelihood, counting close strings as being close
        return sum([datum.output[k]*logsumexp([log(llcounts[r])-log(lo) - distance_factor*distance(r, k) for r in llcounts.keys()]) for k in datum.output.keys()])'''

def make_hypothesis():
    return MyHypothesis(grammar)

def runme(x):
    print "Start: " + str(x)
    return standard_sample(make_hypothesis, make_data, show=False, save_top= "top.pkl", steps=1000000)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample
    #standard_sample(make_hypothesis, make_data, show_skip=9, save_top=False)

    #for running parallel
    from LOTlib.MPI import MPI_map
    args=[[x] for x in range(48)]
    myhyp=set()

    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open("topkaggik.pkl", "wb"))