from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData

from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihood
from LOTlib.Hypotheses.StochasticSimulation import StochasticSimulation
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy
from copy import deepcopy
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib import break_ctrlc
from LOTlib.TopN import TopN
from LOTlib.Miscellaneous import logsumexp,nicelog, Infinity,attrmem
from Levenshtein import distance
from math import log

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="file name of the pickled results", default="out.pkl")
parser.add_option("-d", "--datasize", dest="datasize", type="int", help="number of data points", default=1000)
parser.add_option("-t", "--top", dest="top", type="int", help="top N count of hypotheses from each chain", default=100)
parser.add_option("-s", "--steps", dest="steps", type="int", help="steps for the chainz", default=100000)
parser.add_option("-c", "--chainz", dest="chains", type="int", help="number of chainz :P", default=25)
parser.add_option("--terminals",dest="TERMINALS",help="which terminals are we using? one string")
parser.add_option("--data",dest="DATA",help="what data is seen?")
parser.add_option("-p","--partition", dest="PARTITION", default=False, help="are we running partition MCMC?")
(options, args) = parser.parse_args()
print options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse the input: the terminals and the data seen
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

terminal_file = open(options.TERMINALS,"r")
TERMINALS = []
weights = []
d = []

for line in terminal_file:
    line = line.split(" ")
    TERMINALS.append(line[0])
    weights.append(line[1].strip("\n"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_file = open(options.DATA,"r")

for line in data_file:
    d.append(line.strip("\n"))

def make_data(d=d,size=options.datasize):
    output = {}
    print d
    for val in d:
        output.update({val:size})
    return [FunctionData(input=[],output=output)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@primitive
def prob_():
    return {True:log(.5),False:log(.5)}

TERMINAL_WEIGHT = .1

grammar = Grammar()



# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'sample_uniform_d', ['SET'], 1.) # this requires its own set for terminals
grammar.add_rule('SET', '%s', ['DETTERMINAL'], 1.0)
grammar.add_rule('SET', '%s+%s', ['DETTERMINAL', 'SET'], 1.0)
grammar.add_rule('PROB','prob_()', None,1.0)


# Build up partitions before we have the terminals and strings
# this way, partitions are mainly structural
partitions = []
for t in grammar.enumerate(7):

    t = deepcopy(t) # just make sure it's a copy (may not be necessary)
    for n in t:
        setattr(n, "p_propose", 0.0) # add a tree attribute saying we can't propose
    partitions.append(t)
for part in partitions:
    print part

for t in TERMINALS:
    grammar.add_rule('DETTERMINAL', "'%s'"%t, None, TERMINAL_WEIGHT) # deterministic terminals participate in sets

grammar.add_rule('EXPR', 'if_d', ['PROB','EXPR','EXPR'],1.0)

grammar.add_rule('EXPR', 'cons_d', ['EXPR', 'EXPR'], 1.0/2.0)

for t,w in zip(TERMINALS,weights):
    assert len(t) == 1, "*** Terminals can only be single characters"

    grammar.add_rule('EXPR', "{'%s':0.0}"%t, None, TERMINAL_WEIGHT*float(w))



from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLog

class MyHypothesis(MultinomialLikelihoodLog, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda : %s', **kwargs)
        self.outlier = -100 # for MultinomialLikelihoodLog

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

    def __call__(self, *args, **kwargs):
        d = LOTHypothesis.__call__(self, *args, **kwargs) # now returns a dictionary

        # go through and reformat the keys to have spaces
        #NOTE: requires that terminals are each single chars, see assertion above
        out = dict()
        for k, v in d.items():
            out[' '.join(k)] = v
        return out


def make_hypothesis():
    return MyHypothesis(grammar)

def runme(d,x,datamt,partitions):
    def make_data(d=d,size=datamt):
        output = {}
        for val in d:
            output.update({val:size})
        return [FunctionData(input=[],output=output)]
    print "Start: " + str(x) + " on this many: " + str(datamt)
    if options.PARTITION:
        partitionMCMC(make_data(),partitions)
    else:
        return standard_sample(make_hypothesis, make_data, show=True, N=100, save_top="topModel1.pkl", steps=100000)
def howyoudoin(h):
    doin = False
    for key, values in h().iteritems():
        print key
        if len(key)>=3 and (key[2] == 'e' or key[2]== 'a' or key[2]== 'i'):
            doin = True
    return doin



def partitionMCMC(data,partitions):
    print data
    topn= TopN(N=200, key="posterior_score")
    for p in break_ctrlc(partitions):
        print "Starting on partition ", p

        # Now we have to go in and fill in the nodes that are nonterminals
        v = grammar.generate(deepcopy(p))

        #h0 = MyHypothesis(grammar, value=v)
        h0= make_hypothesis()
        print h0
        for h in break_ctrlc(MHSampler(h0, data, steps=5000, skip=0)):
            # Show the partition and the hypothesis
            print h.posterior_score, p, h, howyoudoin(h)
            topn.add(h)
    return set(topn)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample
    NCHAINS = 25
    #standard_sample(make_hypothesis, make_data, show=True, N=100, save_top="topModel1.pkl", steps=100000)
    from LOTlib.Primitives import *
    import LOTlib.Miscellaneous
    from LOTlib.MPI import MPI_map

    #pass the data, data_amount, and chain number to MPI
    args=[[d,damt, x, partitions] for damt in range(1,options.datasize) for x in range(NCHAINS)]

    myhyp=set()
    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open("out.pkl", "wb"))


