from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

from Grammar import grammar

def make_hypothesis(**kwargs):
    return RationalRulesLOTHypothesis(grammar=grammar, rrAlpha=1.0, **kwargs)
