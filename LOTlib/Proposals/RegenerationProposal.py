from LOTProposal import LOTProposal
from LOTlib.Miscellaneous import lambdaTrue
from copy import copy
from math import log

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

        n, rp, tZ = None, None, None
        for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, do_bv=True, predicate=predicate):
            n = ni

            # re-generate my type from the grammar, and change this functionode
            if self.grammar.is_nonterminal(n.returntype):
                new = self.grammar.generate(n.returntype)
                n.setto(new)
                
            else: pass # do nothing if we aren't returnable from the grammar

            tZ = Z

            rp = resample_p

        newZ = self.grammar.resample_normalizer(newt, predicate=predicate)

        #print "PROPOSED ", newt
        f = (log(rp) - log(tZ))   + newt.log_probability()
        b = (log(rp) - log(newZ)) + t.log_probability()

        if separate_fb:
            return [newt,f, b]
        else:
            return [newt,f-b]
