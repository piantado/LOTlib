from LOTProposal import LOTProposal
import numpy
from LOTlib.Miscellaneous import weighted_sample
from LOTlib.Inference.Proposals import ProposalFailedException


class MixtureProposal(LOTProposal):
    """
            A mixture of proposals, like

            m = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )

            Unnormalized probabilities of each can be specified via probs=[1.0, 2.0]

    """
    def __init__(self, proposals, probs=None):
        self.__dict__.update(locals())

        if probs is None:
            self.probs = numpy.array([1.] * len(proposals))

    def propose_tree(self, t):

        while True:
            try:
                p = weighted_sample(self.proposals, probs=self.probs, log=False)
                ret = p.propose_tree(t)
                return ret
            except ProposalFailedException: # Proposal fails, keep looping
                pass
