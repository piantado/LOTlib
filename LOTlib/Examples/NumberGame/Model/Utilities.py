
from LOTlib.Miscellaneous import logsumexp
import Grammar as G, Hypothesis


# ============================================================================================================
#  Generate number set hypotheses
# ============================================================================================================

def normalizing_constant(hypotheses):
    """Estimate normalizing constant Z by logsumexp(posterior scores for all hypotheses)."""
    return logsumexp([h.posterior_score for h in hypotheses])


def make_h0(grammar=G.grammar, **kwargs):
    """Make initial NumberGameHypothesis."""
    return Hypothesis.NumberGameHypothesis(grammar, **kwargs)


