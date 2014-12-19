from LOTProposal import LOTProposal
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Miscellaneous import lambdaTrue, Infinity, logplusexp, dropfirst
from LOTlib.FunctionNode import FunctionNode
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

    def lp_propose(self, x, y, xZ=None):
        """
                Returns a log probability of starting at x and ending up at y from a regeneration move.
                Any node is a candidate if the trees are identical except for what's below those nodes
                (although what's below *can* be identical!)

                NOTE: This does NOT take into account insert/delete
                NOTE: Not so simple because we must count multiple paths
        """
        RP = -Infinity

        if isinstance(x, FunctionNode) and isinstance(y, FunctionNode) and x.returntype == y.returntype:

            # compute the normalizer
            if xZ is None: xZ = x.sample_node_normalizer()

            # Well we could select x's root to go to Y, but we must recompute y under the current grammar
            RP = logplusexp(RP, log(x.resample_p) - log(xZ) + self.grammar.log_probability(copy(y)) )

            if x.name == y.name and x.args is not None and y.args is not None and len(x.args) == len(y.args):

                # how many kids are not equal, and where was the last?
                mismatch_count, mismatch_index = 0, 0
                for i, xa, ya in zip(xrange(len(x.args)), x.args, y.args):
                    if xa != ya: # checks whole subtree!
                        mismatch_count += 1
                        mismatch_index = i
                    if mismatch_count > 1: break # can't win

                if mismatch_count > 1: # We have to have only selected x,y to regenerate

                    pass

                elif mismatch_count == 1: # we could propose to x, or x.args[mismatch_index], but nothing else (nothing else will fix the mismatch)

                    with BVRuleContextManager(self.grammar, x, recurse_up=False): # recurse, but keep track of bv
                        RP = logplusexp(RP, self.lp_propose(x.args[mismatch_index], y.args[mismatch_index], xZ=xZ))

                else: # identical trees -- we could propose to any, so that's just the tree probability below convolved with the resample p
                    for xi in self.grammar.iterate_subnodes(x):
                        if xi is not x: # but we already counted ourself (NOTE: Must be "is", not ==)
                            RP = logplusexp(RP, log(xi.resample_p) - log(xZ) + self.grammar.log_probability(copy(xi)))
        # print
        # print "X=", x
        # print "Y=", y
        # print "RP=", RP
        return RP