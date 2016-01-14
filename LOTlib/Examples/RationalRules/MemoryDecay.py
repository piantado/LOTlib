"""
A version that cares more about recent data, showing how to use
Hypotheses.DecayedLikelihoodHypothesis.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model import make_data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model import grammar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.Likelihoods.PowerLawDecayed import PowerLawDecayed
from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

class MyHypothesis(PowerLawDecayed, RationalRulesLOTHypothesis):
    """
    Here, DecayedLikelihoodHypothesis provides the decaying functions, RationalRulesLOTHypothesis provides
    everything else.
    """
    def __init__(self, grammar=None, value=None, ll_decay=1.0, args=('x',), **kwargs ):
        # Note: args will just get passed to RationalRulesLOTHypothesis
        RationalRulesLOTHypothesis.__init__(self, grammar, value=value, args=args, **kwargs)

        self.ll_decay = ll_decay # needed here

def make_hypothesis(**kwargs):
    return MyHypothesis(grammar=grammar, rrAlpha=1.0, **kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib.Miscellaneous import q

    # Create an initial hypothesis
    # This is where we set a number of relevant variables -- whether to use RR, alpha, etc.Z
    h0 = MyHypothesis(grammar, ll_decay=1.0, rrAlpha=1.0, args=['x'])

    data = make_data()

    # Run the vanilla sampler. Without steps, it will run infinitely
    # this prints out posterior (posterior_score), prior, likelihood,
    for h in break_ctrlc(MHSampler(h0, data, 10000, skip=100, shortcut_likelihood=False)):
        print h.posterior_score, h.prior, h.likelihood, q(h)

    # This setup requires the *later* data to be upweighted, meaning that hypotheses that get
    # later data wrong should be given lower likelhood. But also with the decay, the overall
    # magnitude of the likelihood decreases.

