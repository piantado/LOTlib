"""
A very simple case of predicate invention, inspired by

T. D. Ullman, N. D. Goodman and J. B. Tenenbaum (2012), Theory learning as stochastic search in the
    language of thought. Cognitive Development.

Here, we invent simple predicates whose value is determined by a set membership (BASE-SET), and express
logical concepts over those predicates. Data is set up to be like magnetism, with positives (pi) and
negatives (ni) that interact with each other but not within groups/

TODO: Let's add another class--the non-magnetic ones!

"""
from LOTlib.FunctionNode import cleanFunctionNodeString
from Data import *
from Grammar import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run mcmc
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run():
    from LOTlib import lot_iter
    from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
    #mp = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
    mp = RegenerationProposal(grammar)

    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    h0 = LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, proposal_function=mp) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary

    from LOTlib.Inference.MetropolisHastings import MHSampler
    for h in lot_iter(MHSampler(h0, data, skip=100)):
        print h.posterior_score, h.likelihood, h.prior, cleanFunctionNodeString(h)
        #print map( lambda d: h(*d.input), data)
        #print "\n"

if __name__ == "__main__":
    run()