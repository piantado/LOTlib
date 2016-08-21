import pickle
import numpy
from LOTlib import break_ctrlc

from Model import *

# Name for output files used in visualisation
MODEL = 'Mixed'

'''
# If we want to run a bigger inference, we can load hypotheses here
with open('HypothesisSpace.pkl', 'r') as f:
    hypotheses = list(pickle.load(f))

print "# Loaded hypotheses: ", len(hypotheses)
'''

# For now, we'll just sample from the prior
hypotheses = set([RationalRulesLOTHypothesis(grammar=grammar, maxnodes=100) for _ in xrange(20)])
for h in hypotheses:
    print h

from LOTlib.DataAndObjects import make_all_objects


objects = make_all_objects(size=['miniature', 'intermediate', 'colossal'],
                           color=['cinnabar', 'viridian', 'cerulean'],
                           shape=['rhombus', 'pentagon', 'dodecahedron'])
data = make_data(dataset=['A', 'B'])
L = [[h.compute_likelihood(dp) for h in hypotheses] for dp in data]
# Store the likelihoods for visualization
with open('Viz/Likelihoods_' + MODEL + '.csv', 'w') as f:
    lines = []
    for l in L:
        lines.extend('\n'.join([str(x) for x in l]))
        lines.extend('\n')
    f.writelines(lines)


# We'll use this to simulate the humans
def human(objs, attr=['color'], value=['cinnabar'], n=200, groups=1):
    nyes = []
    ntrials = []
    for i in xrange(groups):
        for obj in objs:
            ntrials.append(n)
            if eval('obj.' + attr[i]) == value[i]:
                nyes.append(n*0.9)
            else:
                nyes.append(n*0.1)
    return nyes, ntrials

NYes, NTrials = human(objects, attr=['color', 'size'], value=['cinnabar', 'intermediate'], groups=len(data), n=500)

Output = numpy.array([ [1 * h(obj) for h in hypotheses] for obj in objects] * len(data))
# Stash the model responses for vizualization
with open('Viz/Model_' + MODEL + '.csv', 'w') as f:
    f.writelines('\n'.join([','.join([str(r) for r in h]) for h in Output.T]))

GroupLength = [len(objects) for _ in data]

print "# Loaded %s observed rows" % len(NYes)
print "# Organized %s groups" % len(GroupLength)

from LOTlib.Inference.GrammarInference import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt not in ['START']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)

# Stash counts for viz
with open('Viz/Counts_' + MODEL + '.csv', 'w') as f:
    f.writelines('\n'.join([','.join([str(r) for r in h0]) + ',' + ','.join([str(r) for r in h]) for h0, h in zip(counts['BOOL'], counts['PREDICATE'])]))

print "# Computed counts for each hypothesis & nonterminal"

from LOTlib.Inference.GrammarInference.SimpleGrammarHypothesis import SimpleGrammarHypothesis
from LOTlib.Inference.GrammarInference.FullGrammarHypothesis import FullGrammarHypothesis

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

h0 = SimpleGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
# h0 = FullGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)

writ = []
mhs = MHSampler(h0, [], 100, skip=500)
for s, h in break_ctrlc(enumerate(mhs)):


    if isinstance(h, SimpleGrammarHypothesis):
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',BOOLS,' +\
            ','.join([str(x) for x in h.value['BOOL'].value ]) + ',PREDS,' + ','.join([str(x) for x in h.value['PREDICATE'].value ])
    else:
        assert isinstance(h, FullGrammarHypothesis)
        a = str(mhs.acceptance_ratio()) + ',' + str(h.prior) + ',' + str(h.likelihood) +  ',' + \
        str(h.value['alpha'].value[0]) + ',' + str(h.value['beta'].value[0]) + ',' + \
        str(h.value['prior_temperature']) + ',' + str(h.value['likelihood_temperature'])  + ',RULES,' +\
            ','.join([str(x) for x in h.value['rulep']['PREDICATE'].value ])
    print a
    writ.append(a)

with open('Viz/' +MODEL + '.csv', 'w') as f:
    f.writelines('\n'.join(writ))

