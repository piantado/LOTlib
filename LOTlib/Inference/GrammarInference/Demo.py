import pickle

from LOTlib import break_ctrlc

from LearnHypotheses import *

with open('HypothesisSpace.pkl', 'r') as f:
    hypotheses = list(pickle.load(f))

print "# Loaded hypotheses: ", len(hypotheses)

from LOTlib.DataAndObjects import make_all_objects

objects = make_all_objects(size=['small', 'large'], color=['red', 'green'], shape=['square', 'triangle'])

data = make_data()

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
        100,   # 110
        100,   # 101
        100,   # 100
        1,   # 000 TRUE
        1,   # 001
        1,   # 010
        1]   # 011

NTrials = [100]*8

Output = [ [1 * h(obj) for h in hypotheses] for obj in objects]

GroupLength = [8]

print "# Loaded %s observed rows" % len(NYes)
print "# Organized %s groups" % len(GroupLength)

from LOTlib.Inference.GrammarInference.GrammarInference import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt in ['PREDICATE']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)

print "# Computed counts for each hypothesis & nonterminal"

from AlphaBetaGrammar import AlphaBetaGrammarHypothesis
from GrammarHypothesis import GrammarHypothesis
from GrammarLLTHypothesis import GrammarLLTHypothesis
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

h0 = GrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
#h0 = GrammarLLTHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
#h0 = AlphaBetaGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)

mhs = MHSampler(h0, [], 100000, skip=100)
for s, h in break_ctrlc(enumerate(mhs)):
    if isinstance(h0, GrammarHypothesis):
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',' +\
            ','.join([str(x) for x in h.value['PREDICATE'].value ])
    elif isinstance(h0, GrammarLLTHypothesis):
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',' +\
            str(h.value['llt']) + ',' + ','.join([str(x) for x in h.value['rulep']['PREDICATE'].value])
    else:
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',' + \
        str(h.value['alpha'].value[0]) + ',' + str(h.value['beta'].value[0]) + ',' + str(h.value['llt']) + ',' + \
        ','.join([str(x) for x in h.value['rulep']['PREDICATE'].value ])
    print a
