from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import attrmem, Infinity, nicelog
from OptionParser import options
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(size=options.datasize):
    return [FunctionData(input=[],
                         output={'n i k': size, 'h i N': size, 'f a n': size, 'g i f': size, 'm a N': size, 'n a s': size, 'g i k': size, 'k a n': size, 'n i f': size, 'f a g': size, 'g i m': size, 'g i s': size, 's i f': size, 'k i m': size, 'n i m': size, 'g a s': size, 'k a f': size, 'f a s': size, 's i n': size, 's a f': size, 's i k': size, 's i m': size, 'h i m': size, 'h i n': size, 'f a N': size, 'h i k': size, 'k a m': size, 'h i f': size, 'f a m': size, 'g i N': size, 'm i f': size, 'n i s': size, 'k i N': size, 's i N': size, 'n a m': size, 'h i s': size, 'f i s': size, 'k a s': size, 'g a n': size, 'g a m': size, 'h a f': size, 'k i s': size, 'm i n': size, 'k a N': size, 'g a f': size, 'g i n': size, 'k a g': size, 's a n': size, 's a m': size, 'n a f': size, 'n a g': size, 'm i N': size, 's a g': size, 'f i k': size, 'h a N': size, 'f i n': size, 'f i m': size, 'm a s': size, 'g a N': size, 'h a s': size, 'k i f': size, 'n a N': size, 'm i s': size, 's a N': size, 'm i k': size, 'h a g': size, 'm a g': size, 'm a f': size, 'k i n': size, 'h a m': size, 'h a n': size, 'n i N': size, 'f i N': size, 'm a n': size})]



import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)


TERMINAL_WEIGHT = 35

grammar = Grammar()

grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'sample_', ['SET'], 1.0)
grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0/1.5)

grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)
grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)

grammar.add_rule('EXPR', 'if_', ['BOOL', 'EXPR', 'EXPR'], 1./4)
grammar.add_rule('BOOL', 'flip_', [''], 1.)


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
from LOTlib.TopN import TopN

def runme(x):
    print "Start: " + str(x)
    fuckup = TopN(options.top)
    try:
        return standard_sample(make_hypothesis, make_data, show=False, N=options.top, save_top= "top.pkl", steps=options.steps)
    except:
        return fuckup

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample


    #standard_sample(make_hypothesis, make_data, show_skip=9, save_top=False)


    from LOTlib.MPI import MPI_map
    args=[[x] for x in range(40)]
    myhyp=set()

    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open(options.filename, "wb"))