

"""Mixing Insert and Delete Proposers for backward compatibility"""

from LOTlib.Hypotheses.Proposers.DeleteProposer import *
from LOTlib.Hypotheses.Proposers.InsertProposer import *
from LOTlib.Hypotheses.Proposers.RegenerationProposer import *
from LOTlib.Hypotheses.Proposers.MixtureProposer import *
from LOTlib.Miscellaneous import lambdaOne, nicelog

class InsertDeleteRegenerationProposer(MixtureProposer):
    def __init__(self, proposer_weights=[1.0,1.0,1.0]):
        MixtureProposer.__init__(self,proposers=[InsertProposer(),DeleteProposer(), RegenerationProposer()],proposer_weights=proposer_weights)

