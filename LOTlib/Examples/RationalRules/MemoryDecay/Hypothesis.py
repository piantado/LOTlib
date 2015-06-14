from LOTlib.Hypotheses.Likelihoods.PowerLawDecayed import PowerLawDecayed
from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

from LOTlib.Examples.RationalRules.Model.Grammar import grammar

class MyHypothesis(PowerLawDecayed, RationalRulesLOTHypothesis):
    """Define our class solely throuhg inheritance of these two.

    Here, DecayedLikelihoodHypothesis provides the decaying functions, RationalRulesLOTHypothesis provides
    everything else. We just need to pass the initializer to both superclasses.

    """
    def __init__(self, grammar=None, value=None, ll_decay=1.0, args=('x',), **kwargs ):
        # Note: args will just get passed to RationalRulesLOTHypothesis
        RationalRulesLOTHypothesis.__init__(self, grammar, value=value, args=args, **kwargs)

        self.ll_decay = ll_decay # needed here


def make_hypothesis(**kwargs):
    return MyHypothesis(grammar=grammar, rrAlpha=1.0, **kwargs)
