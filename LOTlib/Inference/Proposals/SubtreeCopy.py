from LOTProposal import LOTProposal
from LOTlib.Inference.Proposals import ProposalFailedException
from LOTlib.FunctionNode import NodeSamplingException
from LOTlib.Miscellaneous import lambdaOne, Infinity, logplusexp, dropfirst
from LOTlib.FunctionNode import FunctionNode
from copy import copy
from math import log
from LOTlib.BVRuleContextManager import BVRuleContextManager

class SubtreeCopyProposal(LOTProposal):
    """
            Pick a node and copy it somewhere else
    """














































    def propose_tree(self, t, resampleProbability=lambdaOne):
        """
            Propose to a tree, returning the new tree and the prob. of sampling it.
        """
        
        newt = copy(t)

        try:
            # sample a subnode
            n, lp = newt.sample_subnode(resampleProbability=resampleProbability)
        except NodeSamplingException:
            # If we've been given resampleProbability that can't sample
            raise ProposalFailedException

        # In the context of the parent, resample n according to the grammar
        # We recurse_up in order to add all the parent's rules
        with BVRuleContextManager(self.grammar, n.parent, recurse_up=True): 
            n.setto(self.grammar.generate(n.returntype))
        
        # compute the forward/backward probability    
        f = lp + newt.log_probability()
        b = (log(1.0*resampleProbability(n)) - log(newt.sample_node_normalizer(resampleProbability=resampleProbability))) + t.log_probability()

        return [newt, f-b]
