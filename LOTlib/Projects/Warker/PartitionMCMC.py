from LOTlib.Eval import primitive
from LOTlib.DataAndObjects import FunctionData

from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.LevenshteinLikelihood import StochasticLevenshteinLikelihood
from LOTlib.Hypotheses.Proposers import insert_delete_proposal, ProposalFailedException, regeneration_proposal
import numpy

from LOTlib.Miscellaneous import logsumexp,nicelog, Infinity,attrmem
from Levenshtein import distance
from math import log
# from OptionParser import options
from copy import deepcopy

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
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TERMINAL_WEIGHT = 15

grammar = Grammar()

# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'sample_', ['SET'], 1.)

grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0/2.0)

grammar.add_rule('SET', '"%s"', ['STRING'], 1.0)


# Build up partitions before we have the terminals and strings
# this way, partitions are mainly structural
partitions = []
for t in grammar.enumerate(7):

    t = deepcopy(t) # just make sure it's a copy (may not be necessary)
    for n in t:
        setattr(n, "p_propose", 0.0) # add a tree attribute saying we can't propose
    partitions.append(t)

grammar.add_rule('STRING', '%s%s', ['TERMINAL', 'STRING'], 1.0)
grammar.add_rule('STRING', '%s', ['TERMINAL'], 1.0)

grammar.add_rule('TERMINAL', 'g', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'e', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'k', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 's', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'f', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'n', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'm', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'h', None, TERMINAL_WEIGHT)
grammar.add_rule('TERMINAL', 'N', None, TERMINAL_WEIGHT)




class MyHypothesis(StochasticLikelihood, LOTHypothesis):
#class MyHypothesis(StochasticLevenshteinLikelihood, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, display='lambda : %s', **kwargs)

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc
    #standard_sample(make_hypothesis, make_data, show_skip=9, save_top=False)

    for p in break_ctrlc(partitions):
        print "Starting on partition ", p

        # Now we have to go in and fill in the nodes that are nonterminals
        # We can do this with generate (must pull from github)
        v = grammar.generate(deepcopy(p))

        h0 = MyHypothesis(grammar, value=v)
        size = 100
        data = [FunctionData(input=[],
                             output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]

        for h in break_ctrlc(MHSampler(h0, data, steps=1000, skip=0)):
            # Show the partition and the hypothesis
            print h.posterior_score, p, h

