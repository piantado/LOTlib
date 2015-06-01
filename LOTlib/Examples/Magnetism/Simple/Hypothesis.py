

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal

## Here if we want we can change the proposal function
# mp = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
mp = RegenerationProposal(grammar)

def make_hypothesis(proposal_function=mp, **kwargs):
    return LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, proposal_function=mp, **kwargs) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary

