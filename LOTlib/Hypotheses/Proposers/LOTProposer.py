
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from copy import copy

class LOTProposer(Hypothesis):
    """
            A class of LOT proposals. This wraps calls with copying of the hypothesis
            so that we can implement only propose_t classes for subclasses, that generate trees
    """

    def propose(self, **kwargs):

        while True: # keep trying to propose
            try:
                ret = self.propose_tree(self.value, **kwargs) # don't unpack, since we may return [newt,fb] or [newt,f,b]
                break
            except ProposalFailedException:
                pass

        p = Hypothesis.__copy__(self, value=ret[0])

        ret[0] = p # really make the first a hypothesis, not a tree

        return ret


