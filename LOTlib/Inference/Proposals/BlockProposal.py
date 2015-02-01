from LOTProposal import LOTProposal
import numpy
from LOTlib.Inference.Proposals import ProposalFailedException

class BlockProposal(LOTProposal):
    """
        We do N steps of one proposal, M of the next, etc.
        Default is 10 of each kind
    """

    def __init__(self, proposals, steps=None):
        self.__dict__.update(locals())

        assert len(self.steps) == len(self.proposals)
        if steps is None:
            self.steps = numpy.array([10.] * len(proposals))

        self.idx = 0
        self.n = 0 # how many have we done?

    def propose_tree(self, t):

        if self.n > self.steps[self.idx]:
            self.idx = (self.idx + 1) % len(self.steps) # move to the next kind
            self.n = 0 # And reset the counter

        while True:
            try:
                ret = self.proposals[self.idx].propose_tree(t)
                return ret
            except ProposalFailedException: # Proposal fails, keep looping
                pass
