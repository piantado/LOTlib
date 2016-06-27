
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj

def make_data(n=1, alpha=0.9, dataset=['A']):
    data = []
    if 'A' in dataset:
        data.append([FunctionData(input=[Obj(shape='rhombus', color='cinnabar', size='miniature')], output=False, alpha=alpha),
                FunctionData(input=[Obj(shape='pentagon', color='viridian', size='colossal')], output=True, alpha=alpha)]*n)
    if 'B' in dataset:
        data.append([FunctionData(input=[Obj(shape='rhombus', color='cinnabar', size='miniature')], output=False, alpha=alpha),
                FunctionData(input=[Obj(shape='dodecahedron', color='cerulean', size='intermediate')], output=True, alpha=alpha)]*n)
    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# from LOTlib.DefaultGrammars import SimpleBoolean

# DNF defaultly includes the logical connectives so we need to add predicates to it.
# grammar = SimpleBoolean

from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['PREDICATE'], 1.0)

# Two predicates for checking x's color and shape
# Note: per style, functions in the LOT end in _
grammar.add_rule('PREDICATE', 'is_color_(x, "cinnabar")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_color_(x, "viridian")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_color_(x, "cerulean")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_shape_(x, "rhombus")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_shape_(x, "pentagon")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_shape_(x, "dodecahedron")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_size_(x, "miniature")',  None,  1.0)
grammar.add_rule('PREDICATE', 'is_size_(x, "intermediate")',  None,  1.0)
grammar.add_rule('PREDICATE', 'is_size_(x, "colossal")',  None,  1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

def make_hypothesis(grammar=grammar, **kwargs):
    return RationalRulesLOTHypothesis(grammar=grammar, rrAlpha=1.0, **kwargs)


if __name__ == "__main__":

    from LOTlib.TopN import TopN
    hyps = TopN(N = 1000)

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc
    mhs = MHSampler(make_hypothesis(), make_data(), 1000000, likelihood_temperature = 1., prior_temperature = 1.)

    for samples_yielded, h in break_ctrlc(enumerate(mhs)):
        h.ll_decay = 0.
        hyps.add(h)

    import pickle
    with open('HypothesisSpace.pkl', 'w') as f:
        pickle.dump(hyps, f)
