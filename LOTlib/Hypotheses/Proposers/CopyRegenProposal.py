
"""Copy / Regenerate proposals - choose two nodes of type X and copy
one to the other *OR* choose a node that is duplicated and regenerate
it.

"""

from LOTlib.Hypotheses.Proposers.Regeneration import regeneration_proposal
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import NodeSamplingException
from LOTlib.Miscellaneous import lambdaOne
from copy import copy, deepcopy
from random import random
from math import log

class CopyRegenProposal(object):
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = copy_regen_proposal(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

def give_grammar(grammar,node):
    # Remember: BVRuleContextManager looks at the rule context for
    # generation inside this node, not at the node itself, so we want
    # to consider the node's parent
    with BVRuleContextManager(grammar, node.parent, recurse_up=True):
        g = deepcopy(grammar)
    return g

def copy_regen_proposal(grammar, t, resampleProbability=lambdaOne):
    """Propose, returning the new tree and MH acceptance probability"""

    if random() < 0.5: # copy!
        newt = copy(t)

        # sample the source and then the target conditioned on having the same grammar as the source
        # Note: the two nodes need not be different
        try:
            src, lp_choosing_src_in_old_tree = newt.sample_subnode(resampleProbability=resampleProbability)
            src_grammar = give_grammar(grammar,src)
            good_choice = lambda x: 1.0 if ((give_grammar(grammar,x) == src_grammar) and
                                            (x.returntype == src.returntype)) else 0.0
            target, lp_choosing_target_in_old_tree = newt.sample_subnode(resampleProbability=good_choice)
        except NodeSamplingException:
            raise ProposalFailedException

        lp_target_given_grammar = src_grammar.log_probability(target)

        # set target to be src via a deep copy
        target.setto(deepcopy(src))

        # forward: sample source from old tree, sample target from old tree, copy deterministically
        f = lp_choosing_src_in_old_tree + lp_choosing_target_in_old_tree

        lp_choosing_target_in_new_tree = (log(1.0*resampleProbability(target)) -
                                          log(newt.sample_node_normalizer(resampleProbability=resampleProbability)))

        # backward moves are regeneration: prob to sample target node and regenerate original target tree
        b = lp_choosing_target_in_new_tree + lp_target_given_grammar

        return [newt, f-b]

    else: # regenerate

        return regeneration_proposal(grammar, t, resampleProbability=resampleProbability)
        
if __name__ == "__main__": # test code

    from LOTlib import break_ctrlc
    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

    # We'd probably see better performance on a grammar with fewer
    # distinct types, but this one is a good testbed *because* it's
    # complex (lambdas, etc.)
    from LOTlib.Examples.Magnetism.Simple import grammar, make_data

    class CDHypothesis(BinaryLikelihood, CopyRegenProposal, LOTHypothesis):
        """
        A recursive LOT hypothesis that computes its (pseudo)likelihood using a string edit
        distance
        """
        def __init__(self, **kwargs ):
            LOTHypothesis.__init__(self, grammar, display='lambda x,y: %s', **kwargs)

    def make_hypothesis(**kwargs):
        return CDHypothesis(**kwargs)

    for h in break_ctrlc(MHSampler(make_hypothesis(), data=make_data(n=100), steps=100000)):
        print h.posterior_score, h
