
"""Regenerate proposals - chooses a node of type X and replaces it
with a newly sampled value of type X.

"""

from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import FunctionNode, NodeSamplingException
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Miscellaneous import lambdaOne
from copy import copy
from math import log

class RegenerationProposal(object):
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = regeneration_proposal(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

def regeneration_proposal(grammar, t, resampleProbability=lambdaOne):
    """Propose, returning the new tree and the prob. of sampling it."""

    newt = copy(t)

    try:
        # sample a subnode
        n, lp = newt.sample_subnode(resampleProbability=resampleProbability)
    except NodeSamplingException:
        # If we've been given resampleProbability that can't sample
        raise ProposalFailedException

    assert getattr(n, "resampleProbability", 1.0) > 0.0, "*** Error in propose_tree %s ; %s" % (resampleProbability(t), t)

    # In the context of the parent, resample n according to the grammar
    # We recurse_up in order to add all the parent's rules
    with BVRuleContextManager(grammar, n.parent, recurse_up=True):
        n.setto(grammar.generate(n.returntype))

    # compute the forward/backward probability (i.e. the acceptance distribution)
    f = lp + grammar.log_probability(newt) # p_of_choosing_node_in_old_tree * p_of_new_tree
    b = (log(1.0*resampleProbability(n)) - log(newt.sample_node_normalizer(resampleProbability=resampleProbability)))\
        + grammar.log_probability(t) # p_of_choosing_node_in_new_tree * p_of_old_tree

    return [newt, f-b]
