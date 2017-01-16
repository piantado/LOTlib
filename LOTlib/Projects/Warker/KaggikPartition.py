from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import attrmem, Infinity, nicelog
from copy import deepcopy, copy
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="file name of the pickled results", default="SecondPartitionTests.pkl")
parser.add_option("-d", "--datasize", dest="datasize", type="int", help="number of data points", default=1000)
parser.add_option("-t", "--top", dest="top", type="int", help="top N count of hypotheses from each chain", default=100)
parser.add_option("-s", "--steps", dest="steps", type="int", help="steps for the chainz", default=100000)
parser.add_option("-c", "--chainz", dest="chains", type="int", help="number of chainz :P", default=2)

(options, args) = parser.parse_args()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(size=options.datasize):
    return [FunctionData(input=[],
                         output={'n i k': size, 'h i N': size, 'f a n': size, 'g i f': size, 'm a N': size, 'f a m': size, 'g i k': size, 'k a n': size, 'f a f': size, 'g i n': size, 'g i m': size, 'g i s': size, 's i f': size, 's i n': size, 'n i s': size, 's i m': size, 's i k': size, 'h a N': size, 'f i N': size, 'h i m': size, 'h i n': size, 'h a m': size, 'n i N': size, 'h i k': size, 'f a s': size, 'f i n': size, 'h i f': size, 'n i m': size, 'g i N': size, 'h a g': size, 's i N': size, 'n i n': size, 'f i m': size, 's i s': size, 'h i s': size, 'n a s': size, 'k a s': size, 'f i s': size, 'n i f': size, 'm i n': size, 's a s': size, 'f a g': size, 'k a g': size, 'k a f': size, 's a m': size, 'n a f': size, 'n a g': size, 'm i N': size, 's a g': size, 'f i k': size, 'k a m': size, 'n a n': size, 's a f': size, 'n a m': size, 'm a s': size, 'h a f': size, 'h a s': size, 'n a N': size, 'm i s': size, 's a n': size, 's a N': size, 'm i k': size, 'f a N': size, 'm i m': size, 'm a g': size, 'm a f': size, 'f i f': size, 'k a N': size, 'h a n': size, 'm a n': size, 'm a m': size, 'm i f': size})]


import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)


TERMINAL_WEIGHT = 10
from LOTlib.Primitives import cons_,if_,flip_
from LOTlib.Miscellaneous import *


grammar = Grammar()


grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'sample_', ['SET'], 5.0)

grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0)#downweight the recursion

grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)


grammar.add_rule('EXPR', 'if_', ['BOOL', 'EXPR', 'EXPR'], 1.)#downweight the recursion
grammar.add_rule('BOOL', 'flip_', [''], 1.)

# Enumerate trees to make partitions
partitions = []
for t in grammar.enumerate_at_depth(3, leaves=False):  # 6
    t = deepcopy(t) # just make sure it's a copy (may not be necessary)
    partitions.append(t)

# Take each partition, which doesn't have leaves, and generate leaves, setting
# it to a random generation (fill in the leaves with random hypotheses)
for p in partitions:
    print "# Initializing partition:", p
    print ">>", p
    for n in p.subnodes():
        # set to not resample these
        setattr(n, 'p_propose', 0.0) ## NOTE: Hypothesis proposals must be sensitive to resample_p for this to work!

        # and fill in the missing leaves with a random generation
        for i, a in enumerate(n.args):
            if grammar.is_nonterminal(a):
                n.args[i] = grammar.generate(a)
print "# Initialized %s partitions" % len(partitions)


grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)

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
from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy

from LOTlib.Miscellaneous import logsumexp
#from Levenshtein import distance
from math import log

#class MyHypothesis(StochasticLikelihood, LOTHypothesis):
class MyHypothesis(StochasticLevenshteinLikelihood, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, maxnodes=100,display='lambda : %s', **kwargs)


    #overwrite propose
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                #ret_value, fb = numpy.random.choice([insert_delete_proposal,regeneration_proposal])(self.grammar, self.value, **kwargs)

                # Use a special proposer that won't propose to partition nodes
                ret_value, fb = regeneration_proposal(self.grammar, self.value, resampleProbability=lambda n: getattr(n, "p_propose", 1.0))

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
    '''def compute_single_likelihood(self, datum, distance_factor=100.0):
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"
        llcounts = self.make_ll_counts(datum.input)
        lo = sum(llcounts.values()) # normalizing constant
        # We are going to compute a pseudo-likelihood, counting close strings as being close
        return sum([datum.output[k]*logsumexp([log(llcounts[r])-log(lo) - distance_factor*distance(r, k) for r in llcounts.keys()]) for k in datum.output.keys()])'''

def make_hypothesis():
    return MyHypothesis(grammar)
from LOTlib.TopN import TopN

# def runme(x,datamt):
#     def make_data(size=datamt):
#         return [FunctionData(input=[],
#                              output={'n i k': size, 'h i N': size, 'f a n': size, 'g i f': size, 'm a N': size, 'f a m': size, 'g i k': size, 'k a n': size, 'f a f': size, 'g i n': size, 'g i m': size, 'g i s': size, 's i f': size, 's i n': size, 'n i s': size, 's i m': size, 's i k': size, 'h a N': size, 'f i N': size, 'h i m': size, 'h i n': size, 'h a m': size, 'n i N': size, 'h i k': size, 'f a s': size, 'f i n': size, 'h i f': size, 'n i m': size, 'g i N': size, 'h a g': size, 's i N': size, 'n i n': size, 'f i m': size, 's i s': size, 'h i s': size, 'n a s': size, 'k a s': size, 'f i s': size, 'n i f': size, 'm i n': size, 's a s': size, 'f a g': size, 'k a g': size, 'k a f': size, 's a m': size, 'n a f': size, 'n a g': size, 'm i N': size, 's a g': size, 'f i k': size, 'k a m': size, 'n a n': size, 's a f': size, 'n a m': size, 'm a s': size, 'h a f': size, 'h a s': size, 'n a N': size, 'm i s': size, 's a n': size, 's a N': size, 'm i k': size, 'f a N': size, 'm i m': size, 'm a g': size, 'm a f': size, 'f i f': size, 'k a N': size, 'h a n': size, 'm a n': size, 'm a m': size, 'm i f': size})]
#     print "Start: " + str(x) + " on this many: " + str(datamt)
#     messup = TopN(options.top)
#     try:
#         return standard_sample(make_hypothesis, make_data, show=False, N=options.top, save_top="topkaggik.pkl", steps=options.steps)
#     except:
#         return messup

def runparts(size, x, p):
    #problem: right now only recording last partition, never saving from others.
    print "Start: " + str(x) + " on this many: " + str(size)
    try:
        #make new TopN for each data amount
        topn= TopN(N=200, key="posterior_score")
        print "Starting on partition ", p

        # Now we have to go in and fill in the nodes that are nonterminals
        # We can do this with generate
        v = grammar.generate(copy(p))

        h0 = MyHypothesis(grammar, value=v)
        data = [FunctionData(input=[],
                         output={'n i k': size, 'h i N': size, 'f a n': size, 'g i f': size, 'm a N': size, 'f a m': size, 'g i k': size, 'k a n': size, 'f a f': size, 'g i n': size, 'g i m': size, 'g i s': size, 's i f': size, 's i n': size, 'n i s': size, 's i m': size, 's i k': size, 'h a N': size, 'f i N': size, 'h i m': size, 'h i n': size, 'h a m': size, 'n i N': size, 'h i k': size, 'f a s': size, 'f i n': size, 'h i f': size, 'n i m': size, 'g i N': size, 'h a g': size, 's i N': size, 'n i n': size, 'f i m': size, 's i s': size, 'h i s': size, 'n a s': size, 'k a s': size, 'f i s': size, 'n i f': size, 'm i n': size, 's a s': size, 'f a g': size, 'k a g': size, 'k a f': size, 's a m': size, 'n a f': size, 'n a g': size, 'm i N': size, 's a g': size, 'f i k': size, 'k a m': size, 'n a n': size, 's a f': size, 'n a m': size, 'm a s': size, 'h a f': size, 'h a s': size, 'n a N': size, 'm i s': size, 's a n': size, 's a N': size, 'm i k': size, 'f a N': size, 'm i m': size, 'm a g': size, 'm a f': size, 'f i f': size, 'k a N': size, 'h a n': size, 'm a n': size, 'm a m': size, 'm i f': size})]

        for h in break_ctrlc(MHSampler(h0, data, steps=options.steps, trace=False)):
            # print "\t", h.posterior_score, h
            topn.add(h)

        return size, set(topn)

    except Exception as e:
        print "*** Exception ignored: ", e
        #if we fail, we can return a blank TopN
        return size, set()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    import itertools
    from LOTlib.Inference.Samplers.StandardSample import standard_sample
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc

    from LOTlib.MPI import MPI_map
    args=list(itertools.product(range(1, options.datasize+2,100),
                                range(options.chains),
                                partitions))
    all_hypotheses=set()

    for ndata, top in MPI_map(runparts, args):
        for h in top:
            print ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata, h
            all_hypotheses.add(h)

    import pickle
    pickle.dump(all_hypotheses, open(options.filename, "wb"))




