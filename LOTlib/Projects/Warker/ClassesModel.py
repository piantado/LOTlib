from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData
import random
from LOTlib.Miscellaneous import logsumexp,nicelog, Infinity,attrmem

from OptionParser import options


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(size=options.datasize):

    return [FunctionData(input=[],
                         output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]




import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Grammar
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Let's make the natural classes
@primitive
def bilabial_():
    return "pbmw"

@primitive
def frontvowels_():
    return "ie"

@primitive
def fricatives_():
    return "fvtszh"
@primitive
def alveolar_():
    return "tdsznlr"
@primitive
def glottal_():
    return "h"
@primitive
def palatal_():
    return "j"

@primitive
def velars_():
    return"kgN"

@primitive
def bLNNstops():
    return "pb"
@primitive
def vNNstops():
    return "ptk"
@primitive
def voicedNNstops():
    return "bdg"
@primitive
def labfric():
    return "fv"
@primitive
def vl():
    return "f"
@primitive
def vfrics():
    return "fTsSh"
@primitive
def alvfrics():
    return "sz"
@primitive
def glotfrics():
    return "h"
@primitive
def stops():
    return "pbtdkgmnN"
@primitive
def obstruents():
    return "pbtdkgfvTszSh"
@primitive
def vN():
    return "mnN"
@primitive
def valvliqu():
    return "lr"
@primitive
def lateralliqu():
    return "l"
@primitive
def retroflex():
    return "r"
@primitive
def bilNS():
    return "m"
@primitive
def alvNS():
    return "n"
@primitive
def velarNS():
    return "N"
@primitive
def glides():
    return "wj"
@primitive
def voicedvelar():
    return "gN"
@primitive
def vvelar():
    return "k"
@primitive
def sonorants():
    return "mnNlrwjiIueoaA"





@primitive
def strintersection(s1, s2):
  out = ""
  for c in s1:
    if c in s2 and not c in out:
      out += c
  return out

@primitive

def strunion(s1, s2):
    out = s1
    for c in s2:
        if not c in out:
            out += c
    return out

@primitive
def strdifference(s1, s2):
  out = ""
  for c in s1:
    if c not in s2 and not c in out:
      out += c
  return out



TERMINAL_WEIGHT = 250

grammar = Grammar()

grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'sample_', ['SET'], 1.0)
grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0/2.0)
grammar.add_rule('EXPR','"%s"', ['TERMINAL'],1.0)

#clean string version
'''grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)
grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)'''

#set operations
grammar.add_rule('SET', 'strunion', ['SET', 'SET'], 1.0/10.)
grammar.add_rule('SET', 'strintersection', ['SET', 'SET'], 1.0/10.)
grammar.add_rule('SET', 'strdifference', ['SET', 'SET'], 1.0/10.)

#sets go to the natural classes
grammar.add_rule('SET', 'bilabial_()', None, 1.0)
grammar.add_rule('SET', 'frontvowels_()', None, 1.0)
grammar.add_rule('SET', 'fricatives_()', None, 1.0)
grammar.add_rule('SET', 'alveolar_()', None, 1.0)
grammar.add_rule('SET', 'glottal_()', None, 1.0)
grammar.add_rule('SET', 'palatal_()', None, 1.0)
grammar.add_rule('SET', 'bLNNstops()', None, 1.0)
grammar.add_rule('SET', 'vNNstops()', None, 1.0)
grammar.add_rule('SET', 'voicedNNstops()', None, 1.0)
grammar.add_rule('SET', 'labfric()', None, 1.0)
grammar.add_rule('SET', 'vl()', None, 1.0)
grammar.add_rule('SET', 'vfrics()', None, 1.0)
grammar.add_rule('SET', 'alvfrics()', None, 1.0)
grammar.add_rule('SET', 'glotfrics()', None, 1.0)
grammar.add_rule('SET', 'stops()', None, 1.0)
grammar.add_rule('SET', 'obstruents()', None, 1.0)
grammar.add_rule('SET', 'vN()', None, 1.0)
grammar.add_rule('SET', 'valvliqu()', None, 1.0)
grammar.add_rule('SET', 'lateralliqu()', None, 1.0)
grammar.add_rule('SET', 'retroflex()', None, 1.0)
grammar.add_rule('SET', 'bilNS()', None, 1.0)
grammar.add_rule('SET', 'alvNS()', None, 1.0)
grammar.add_rule('SET', 'velarNS()', None, 1.0)
grammar.add_rule('SET', 'glides()', None, 1.0)
grammar.add_rule('SET', 'voicedvelar()', None, 1.0)
grammar.add_rule('SET', 'vvelar()', None, 1.0)
grammar.add_rule('SET', 'sonorants()', None, 1.0)

# my terminal sounds
grammar.add_rule('TERMINAL', 'g', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'e', None, TERMINAL_WEIGHT)
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

def runme(x,datamt):
    def make_data(size=datamt):
        return [FunctionData(input=[],
                             output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]

    print "Start: " + str(x) + " on this many: " + str(datamt)
    return standard_sample(make_hypothesis, make_data, show=False, N=options.top, save_top="topModel1.pkl", steps=options.steps)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, show_skip=9, save_top=False)

    from LOTlib.MPI import MPI_map
    '''args=[[x, d] for d in range(1, options.datasize+2,10) for x in range(options.chains)]
    myhyp=set()

    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open(options.filename, "wb"))'''
from LOTlib.Miscellaneous import *
from LOTlib.Primitives import *

