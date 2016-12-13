from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData

from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihood
from LOTlib.Hypotheses.StochasticSimulation import StochasticSimulation
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy

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
(options, args) = parser.parse_args()
print options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_data(size=options.datasize):
        return [FunctionData(input=[],
                             output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]


import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse the input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f = open(options.TERMINALS,"r")
WEIGHTED = f.readline().strip("\n")
TERMINALS = f.readline().strip("\n")
weights = []
if WEIGHTED =="True":
    weights = f.readline().split()

d = open(options.DATA,"r").readline().strip("\n").split(",")
def make_data(d=d,size=options.datasize):
    output = {}
    for val in d:
        output.update({val:size})
    return [FunctionData(input=[],output=output)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TERMINAL_WEIGHT = .1

grammar = Grammar()



# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'sample_uniform_d', ['SET'], 1.) # this requires its own set for terminals
grammar.add_rule('SET', '%s', ['DETTERMINAL'], 1.0)
grammar.add_rule('SET', '%s+%s', ['DETTERMINAL', 'SET'], 1.0)
for t in TERMINALS:
    grammar.add_rule('DETTERMINAL', "'%s'"%t, None, TERMINAL_WEIGHT) # deterministic terminals participate in sets

grammar.add_rule('EXPR', 'cons_d', ['EXPR', 'EXPR'], 1.0/2.0)

for t,w in zip(TERMINALS,weights):
    assert len(t) == 1, "*** Terminals can only be single characters"
    if not WEIGHTED:
        grammar.add_rule('EXPR', "{'%s':0.0}"%t, None, TERMINAL_WEIGHT)
    else:
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

def runme(x,datamt):
    def make_data(d=d,size=options.datasize):
        output = {}
        for val in d:
            output.update({val:size})
        return [FunctionData(input=[],output=output)]
    print "Start: " + str(x) + " on this many: " + str(datamt)
    return standard_sample(make_hypothesis, make_data, show=True, N=100, save_top="topModel1.pkl", steps=100000)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    NCHAINS = 1

    standard_sample(make_hypothesis, make_data, show_skip=9, save_top=False)

    '''from LOTlib.MPI import MPI_map
    args=[[x, d] for d in range(1,10) for x in range(NCHAINS)]

    myhyp=set()
    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open("out.pkl", "wb"))'''


