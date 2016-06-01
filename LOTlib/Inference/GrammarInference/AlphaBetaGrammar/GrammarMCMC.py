import pickle
from LearnHypotheses import *

with open('HypothesisSpace.pkl', 'r') as f:
    hypotheses = list(pickle.load(f))

print "# Loaded hypotheses: ", len(hypotheses)

objects = ['RED_SQUARE_LARGE', 'RED_SQUARE_SMALL', 'RED_TRIANGLE_LARGE', 'RED_TRIANGLE_SMALL',
           'GREEN_TRIANGLE_SMALL', 'GREEN_TRIANGLE_LARGE', 'GREEN_SQUARE_SMALL', 'GREEN_SQUARE_LARGE']

data = [FunctionData(input = ["GREEN_TRIANGLE_SMALL"], output = 1, alpha = 0.9),
        FunctionData(input = ["RED_SQUARE_LARGE"], output = 0, alpha = 0.9)]

L = [[h.compute_likelihood(data) for h in hypotheses]]
'''
Red      = 1__
Square   = _1_
Large    = __1
Green    = 0__
Triangle = _0_
Small    = __0
'''

NYes = [100,   # 111 FALSE
        1,   # 110
        1,   # 101
        1,   # 100
        1,   # 000 TRUE
        1,   # 001
        1,   # 010
        1]   # 011

NTrials = [100]*8

Output = [ [1 * h(obj) for h in hypotheses] for obj in objects]

GroupLength = [8]

print "# Loaded %s observed rows" % len(NYes)
print "# Organized %s groups" % len(GroupLength)

from LOTlib.GrammarInference.GrammarInference import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt in ['PREDICATE']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)

print "# Computed counts for each hypothesis & nonterminal"

from AlphaBetaGrammar import AlphaBetaGrammarMH
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

h0 = AlphaBetaGrammarMH(counts, hypotheses, L, GroupLength, prior_offset, NYes, NTrials, Output, scale=600, step_size=0.5)
mhs = MHSampler(h0, [], 10000)
for s, h in enumerate(mhs):
    if s % 100 == 0:
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) + ',' + ','.join([str(x) for x in h.value['PREDICATE']])
        print a
        a += ',' + str(h.alpha) + ',' + str(h.beta) + ',' + str(h.llt)
        print str(h.alpha) + ',' + str(h.beta) + ',' + str(h.llt)
