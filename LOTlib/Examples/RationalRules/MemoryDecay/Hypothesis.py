from LOTlib.Hypotheses.DecayedLikelihoodHypothesis import DecayedLikelihoodHypothesis
from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

class MyHypothesis(DecayedLikelihoodHypothesis, RationalRulesLOTHypothesis):
    """Define our class solely throuhg inheritance of these two.

    Here, DecayedLikelihoodHypothesis provides the decaying functions, RationalRulesLOTHypothesis provides
    everything else. We just need to pass the initializer to both superclasses.

    """
    def __init__(self, grammar, value=None, ll_decay=0.0, args=('x',), **kwargs ):
        # Note: args will just get passed to RationalRulesLOTHypothesis
        DecayedLikelihoodHypothesis.__init__(self, value=None, ll_decay=ll_decay)
        RationalRulesLOTHypothesis.__init__(self, grammar, value=value, args=args, **kwargs)
