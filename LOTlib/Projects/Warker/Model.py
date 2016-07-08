from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(size=100):
    return [FunctionData(input=[],
                         #output={'b i m': size, 'b o p': size})]
                         #output = {'n a f': 100, 'f a k': 100, 's i s': 100, 'n a s': 100, 'f a f': 100, 'k i n': 100, 'k i m': 100, 'n i m': 100, 's a f': 100, 'k i g': 100, 'f a s': 100, 's i m': 100, 'f a m': 100, 'n i s': 100, 'f a n': 100, 'n a m': 100, 'f i s': 100, 'g a n': 100, 'g a m': 100, 'n i f': 100, 'm a f': 100, 's a s': 100, 'g a f': 100, 's a n': 100, 's a m': 100, 's a k': 100, 'f i g': 100, 'g a s': 100, 'n a k': 100, 'n a n': 100, 'f i m': 100, 'm a s': 100, 'k i s': 100, 's i f': 100, 'k i f': 100, 'm i s': 100, 'm i m': 100, 'm i n': 100, 'f i f': 100, 'm a k': 100, 'm i f': 100})]
                         output = {'f e N': 100, 'k e s': 100, 'h e s': 100, 'm e s': 100, 'g e N': 100, 'n e m': 100, 'h e k': 100, 'm e m': 100, 'm e g': 100, 'h e g': 100, 'n e N': 100, 'm e n': 100, 'h e m': 100, 'h e n': 100, 'f e k': 100, 'f e n': 100, 'g e n': 100, 'g e m': 100, 'f e m': 100, 'g e k': 100, 'n e s': 100, 'n e g': 100, 'g e g': 100, 'f e g': 100, 'f e s': 100, 'k e N': 100, 'k e n': 100, 'm e k': 100, 'n e n': 100, 'm e N': 100, 'g e s': 100, 'n e k': 100, 'k e m': 100})]






import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q
from LOTlib.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TERMINAL_WEIGHT = 15

grammar = Grammar()

# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'sample_', ['SET'], 1.)

grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0/2.0)

grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)
grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)




grammar.add_rule('TERMINAL', 'g', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'e', None, TERMINAL_WEIGHT)
#grammar.add_rule('TERMINAL', 'a', None, TERMINAL_WEIGHT)
#grammar.add_rule('TERMINAL', 'i', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'k', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 's', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'f', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'n', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'm', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'h', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'N', None, TERMINAL_WEIGHT)

'''grammar.add_rule('TERMINAL', 'b', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'm', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'p', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'i', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'o', None, TERMINAL_WEIGHT)'''




from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy

from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log

class MyHypothesis(StochasticFunctionLikelihood, LOTHypothesis):
#Levenshtein distance allows us to accept a bit of noise when calculating our likelihood for our proposals
#class MyHypothesis(StochasticLevenshteinLikelihood, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda : %s', **kwargs)


    # we can get stuck pretty easily without this insert/delete ability.
    # This will allow an insertion/deletion in the generated FunctionNodes

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
    return standard_sample(make_hypothesis, make_data, show=False, save_top="top.pkl", steps=10000)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    #standard_sample(make_hypothesis, make_data, show_skip=9, save_top="top.pkl")

    #for running parallel
    from LOTlib.MPI import MPI_map
    args=[[x] for x in range(8)]
    myhyp=set()

    for top in MPI_map(runme, args):
        myhyp.update(top)

    import pickle
    pickle.dump(myhyp, open("tophyp.pkl", "wb"))


