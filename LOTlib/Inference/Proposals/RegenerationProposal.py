from LOTProposal import LOTProposal
from LOTlib.Miscellaneous import lambdaTrue
from copy import copy
from math import log
from LOTlib.BVRuleContextManager import BVRuleContextManager

class RegenerationProposal(LOTProposal):
    """
            Propose to a tree by sampling a node at random and regenerating
    """

    def propose_tree(self, t, separate_fb=False, predicate=lambdaTrue):
        """
                If separate_fb=True -- return [newt, f, b], instead of [newt,f-b]
                NOTE: This used to copy but no longer!
        """
        
        newt = copy(t)

        # sample a subnode
        n, lp = newt.sample_subnode(predicate=predicate)
        
        # In the context of the parent, resample n according to the grammar
        # We recurse_up in order to add all the parent's rules
        with BVRuleContextManager(self.grammar, n.parent, recurse_up=True): 
            n.setto(self.grammar.generate(n.returntype))
        
        # compute the forward/backward probability    
        f = lp + newt.log_probability()
        b = (log(n.resample_p) - log(newt.sample_node_normalizer(predicate=predicate))) + t.log_probability()

        if separate_fb:
            return [newt, f, b]
        else:
            return [newt,f-b]

            

