import pickle
import numpy
from LOTlib import break_ctrlc


from Model import *

'''
# If we want to run a bigger inference, we can load hypotheses here
with open('HypothesisSpace.pkl', 'r') as f:
    hypotheses = list(pickle.load(f))

print "# Loaded hypotheses: ", len(hypotheses)
'''

# For now, we'll just sample from the prior
hypotheses = set([RationalRulesLOTHypothesis(grammar=grammar, maxnodes=100) for _ in xrange(10000)])

from LOTlib.DataAndObjects import make_all_objects

objects = make_all_objects(size=['small', 'large'], color=['red', 'green'], shape=['square', 'triangle'])

data = make_data()

L = [[h.compute_likelihood(data) for h in hypotheses]]

# We'll use this to simulate the human
def human(obj):
    if obj.shape == 'square':
        return 100
    else:
        return 1

NYes = [human(o) for o in objects]

NTrials = [100]*len(objects)


Output = numpy.array([ [1 * h(obj) for h in hypotheses] for obj in objects])

GroupLength = [8]

print "# Loaded %s observed rows" % len(NYes)
print "# Organized %s groups" % len(GroupLength)

from LOTlib.Inference.GrammarInference import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt in ['PREDICATE']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)


print "# Computed counts for each hypothesis & nonterminal"

from LOTlib.Inference.GrammarInference.SimpleGrammarHypothesis import SimpleGrammarHypothesis
from LOTlib.Inference.GrammarInference.FullGrammarHypothesis import FullGrammarHypothesis

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

#h0 = SimpleGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
h0 = FullGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)

mhs = MHSampler(h0, [], 100000, skip=500)
for s, h in break_ctrlc(enumerate(mhs)):


    if isinstance(h, SimpleGrammarHypothesis):
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',RULES,' +\
            ','.join([str(x) for x in h.value['PREDICATE'].value ])
    else:
        assert isinstance(h, FullGrammarHypothesis)
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',' + \
        str(h.value['alpha'].value[0]) + ',' + str(h.value['beta'].value[0]) + ',' + \
        str(h.value['prior_temperature']) + ',' + str(h.value['likelihood_temperature'])  + ',RULES,' +\
            ','.join([str(x) for x in h.value['rulep']['PREDICATE'].value ])
    print a
