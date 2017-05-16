import LOTlib
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
from LOTlib.Hypotheses.Likelihoods.StringLikelihoods import MonkeyNoiseLikelihood
from random import shuffle
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib import break_ctrlc
from LOTlib.TopN import TopN
from LOTlib.Flip import compute_outcomes
from LOTlib.Miscellaneous import display_option_summary, sample1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="file name of the pickled results", default="out.pkl")
parser.add_option("-d", "--datasize", dest="datasize", type="int", help="number of data points", default=100)
parser.add_option("-t", "--top", dest="top", type="int", help="top N count of hypotheses from each chain", default=100)
parser.add_option("-s", "--steps", dest="steps", type="int", help="steps for the chainz", default=1000)
parser.add_option("-c", "--chainz", dest="chains", type="int", help="number of chainz :P", default=25)
parser.add_option("--terminals",dest="TERMINALS",help="which terminals are we using? one string", default = 'Terminals/exponly.txt')
parser.add_option("--data",dest="DATA",help="what data is seen?", default = "Data/firstDATA.txt")
(options, args) = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

terminal_file = open(options.TERMINALS,"r")
TERMINALS = []
weights = []
d = []

for line in terminal_file:
    line = line.split(" ")
    TERMINALS.append(line[0])
    weights.append(line[1].strip("\n"))

DATA_STRINGS = []
with open(options.DATA,"r") as data_file:
    for line in data_file:
        DATA_STRINGS.append(line.strip("\n"))

def make_data(n):
    return [FunctionData(input=[], output={val : n for val in DATA_STRINGS}, alpha=0.999)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

TERMINAL_WEIGHT = 3.0

grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'C.uniform_sample', ['SET'], 1.0) # this requires its own set for terminals
grammar.add_rule('SET', '%s', ['DETTERMINAL'], 1.0)
grammar.add_rule('SET', '%s+%s', ['DETTERMINAL', 'SET'], 1.0)
grammar.add_rule('BOOL','C.flip(0.5)', None,1.0)
grammar.add_rule('EXPR', '(%s if %s else %s)', ['EXPR','BOOL','EXPR'],1.0)
grammar.add_rule('EXPR', 'strcons_(%s, %s, sep=" ")', ['EXPR', 'EXPR'], 1.0)


for w,t in zip(weights, TERMINALS):
    assert len(t) == 1, "*** Terminals can only be single characters"
    grammar.add_rule('DETTERMINAL', "'%s'"%t, None, TERMINAL_WEIGHT/len(TERMINALS)) # deterministic terminals participate in sets
    grammar.add_rule('EXPR', "'%s'"%t,        None, TERMINAL_WEIGHT*float(w)/len(TERMINALS)) # otherwise terminals are EXPR


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLog

class MyHypothesis(MonkeyNoiseLikelihood, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda C : %s', maxnodes=200, **kwargs)
        # self.outlier = -100 # for MultinomialLikelihoodLog
        self.alphabet_size = len(TERMINALS)

    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = sample1([insert_delete_proposal,regeneration_proposal])(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

    def __call__(self, *args, **kwargs):
        return compute_outcomes(lambda *args: LOTHypothesis.__call__(self, *args, **kwargs), maxcontext=100000)


def make_hypothesis():
    return MyHypothesis()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI running
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def runme(chain, dataamt):

    if LOTlib.SIG_INTERRUPTED: return ()

    data = make_data(dataamt)

    tn = TopN(options.top)

    h0 = make_hypothesis()
    for h in break_ctrlc(MHSampler(h0, data, steps=options.steps, skip=0)):
        # print h.posterior_score, h.prior, h.likelihood, h
        h.likelihood_per_data = h.likelihood/dataamt
        tn.add(h)

    return tn

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.MPI import MPI_unorderedmap, is_master_process

    if is_master_process():
        display_option_summary(options)

    #pass the data, data_amount, and chain number to MPI
    args=[[chain, damt] for damt in range(1, options.datasize) for chain in range(options.chains)]
    shuffle(args)

    hypotheses=set()
    for top in MPI_unorderedmap(runme, args):
        hypotheses.update(top)

        for h in top:
             print h.posterior_score, h.prior, h.likelihood, h.likelihood_per_data, h


    import pickle
    pickle.dump(hypotheses, open("out.pkl", "wb"))


