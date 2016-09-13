"""Regeneration proposal - choose a node of type X and replace it with
a newly sampled value of type X.

"""

from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import NodeSamplingException
from LOTlib.Hypotheses.Proposers import *
from LOTlib.Miscellaneous import lambdaOne, logsumexp
from LOTlib.Subtrees import least_common_difference
from copy import copy
from math import log

class RegenerationProposal(object):
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while not ret_value: # keep trying to propose
            try:
                ret_value, fb =  regeneration_proposal(self.grammar, self.value, **kwargs)
            except ProposalFailedException:
                pass
        ret = self.__copy__(value=ret_value)
        return ret, fb

def regeneration_proposal(grammar, tree, resampleProbability=lambdaOne):
    t = regenerate(grammar,tree,resampleProbability)
    fb = regeneration_fb(grammar,tree,t,resampleProbability)
    return t,fb

def regenerate(grammar, t, resampleProbability=lambdaOne):
    """Propose, returning the new tree"""
    new_t = copy(t)

    try: # to sample a subnode
        n, lp = new_t.sample_subnode(resampleProbability=resampleProbability)
    except NodeSamplingException: # when no nodes can be sampled
        raise ProposalFailedException

    # In the context of the parent, resample n according to the
    # grammar. recurse_up in order to add all the parent's rules
    with BVRuleContextManager(grammar, n.parent, recurse_up=True):
        n.setto(grammar.generate(n.returntype))
    return new_t

def regeneration_probability(grammar, t1, t2, resampleProbability=lambdaOne, recurse=True):
    chosen_node1 , chosen_node2 = least_common_difference(t1,t2)

    lps = []
    if chosen_node1 is None: # any node in the tree could have been regenerated
        for node in t1:
            lp_of_choosing_node = t1.sampling_log_probability(node,resampleProbability=resampleProbability)
            with BVRuleContextManager(grammar, node.parent, recurse_up=True):
                lp_of_generating_tree = grammar.log_probability(node)
            lps += [lp_of_choosing_node + lp_of_generating_tree]
    else: # we have a specific path up the tree
        while chosen_node1:
            lp_of_choosing_node = t1.sampling_log_probability(chosen_node1,resampleProbability=resampleProbability)
            with BVRuleContextManager(grammar, chosen_node2.parent, recurse_up=True):
                lp_of_generating_tree = grammar.log_probability(chosen_node2)
            lps += [lp_of_choosing_node + lp_of_generating_tree]
            if recurse:
                chosen_node1 = chosen_node1.parent
                chosen_node2 = chosen_node2.parent
            else:
                chosen_node1 = None

    return logsumexp(lps)

def regeneration_fb(grammar, t1, t2, resampleProbability=lambdaOne):
    return (regeneration_probability(grammar,t1,t2,resampleProbability) -
            regeneration_probability(grammar,t2,t1,resampleProbability))

if __name__ == "__main__": # test code

    from LOTlib.Examples.Magnetism.Simple import grammar, make_data
    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    class CRHypothesis(BinaryLikelihood, RegenerationProposal, LOTHypothesis):
        def __init__(self, **kwargs ):
            LOTHypothesis.__init__(self, grammar, display='lambda x,y: %s', **kwargs)

    def make_hypothesis(**kwargs):
        return CRHypothesis(**kwargs)

    standard_sample(make_hypothesis, make_data, save_top=False)
