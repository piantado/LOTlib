"""
A very simple case of predicate invention, inspired by

T. D. Ullman, N. D. Goodman and J. B. Tenenbaum (2012), Theory learning as stochastic search in the
    language of thought. Cognitive Development.

Here, we invent simple predicates whose value is determined by a set membership (BASE-SET), and express
logical concepts over those predicates. Data is set up to be like magnetism, with positives (pi) and
negatives (ni) that interact with each other but not within groups.

This is simple because there's only two types of things, and you observe all interactions. See
ComplexMagnetism.py for a more complex case...

"""
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Data import data
from Grammar import grammar
from Utilities import make_h0


def run():
    from LOTlib import break_ctrlc
    from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
    from LOTlib.Inference.MetropolisHastings import MHSampler

    # mp = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
    mp = RegenerationProposal(grammar)

    # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary
    h0 =  make_h0(proposal_function=mp) #LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, proposal_function=mp)

    for h in break_ctrlc(MHSampler(h0, data, 4000000, skip=100)):
        print h.posterior_score, h.likelihood, h.prior,  cleanFunctionNodeString(h)
        print map( lambda d: h(*d.input), data)


if __name__ == "__main__":
    run()