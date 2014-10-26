
from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis
from Data import data
from Grammar import grammar, DNF


def make_h0(value=None):
    return RationalRulesLOTHypothesis(grammar=DNF, value=value, rrAlpha=1.0)


if __name__ == "__main__":
    
    from LOTlib.Inference.MetropolisHastings import MHSampler
    for h in MHSampler(make_h0(), data):
        print h