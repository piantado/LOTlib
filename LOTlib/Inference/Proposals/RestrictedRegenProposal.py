# -*- coding: utf-8 -*-
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal

class RestrictedRegenProposal(RegenerationProposal):
    """
    A standard regen proposal but with a restriction on which types are valid to
    regenerate. Specify *EITHER* a whitelist (of valid types) or a blacklist (of
    invalid types)
    """
    def __init__(self, grammar, whitelist=None, blacklist=None, **kwargs):
        self.__dict__.update(locals())
        self.regen_proposal = RegenerationProposal(grammar, **kwargs)

    def propose_tree(self, tree):
        def isvalid(node):
            if self.whitelist:
                return node.returntype in self.whitelist
            elif self.blacklist:
                return node.returntype not in self.blacklist
            else:
                return True

        return self.regen_proposal.propose_tree(tree, resampleProbability=isvalid)

