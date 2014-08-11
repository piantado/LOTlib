from LOTProposal import LOTProposal
import numpy
from LOTlib.Miscellaneous import weighted_sample

class MixtureProposal(LOTProposal):
    """
            A mixture of proposals, like

            m = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )

            Probabilities of each can be specified

    """
    def __init__(self, proposals, probs=None):
        #print proposals
        self.__dict__.update(locals())

        if probs is None:
            self.probs = numpy.array( [1.] * len(proposals) )

    def propose_tree(self, t):
        p = weighted_sample(self.proposals, probs=self.probs, log=False)

        return p.propose_tree(t)
