import pickle
import numpy
import random

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up some logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Miscellaneous import setup_directory, qq

LOG = "log"
setup_directory(LOG) # make a directory for ourselves

# Make sure we print the entire matrix
numpy.set_printoptions(threshold=numpy.inf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses.pkl', 'r') as f:
# with open('hypotheses/hypotheses-1.pkl', 'r') as f:
    concept2hypotheses = pickle.load(f)

hypotheses = set()
for hset in concept2hypotheses.values():
    hypotheses.update(hset)

print "# Loaded %s hypotheses" % len(hypotheses)
with open(LOG+"/hypotheses.txt", 'w') as f:
    for i, h in enumerate(hypotheses):
        print >>f, i, qq(h)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We will map tuples of concept-list, set, response to counts.
import pandas
import math
from collections import Counter
human_data = pandas.read_csv('HumanData/TurkData-Accuracy.txt', sep='\t', low_memory=False, index_col=False)
human_yes, human_no = Counter(), Counter()
for r in xrange(human_data.shape[0]): # for each row
    cl = build_conceptlist(human_data['concept'][r], human_data['list'][r])
    rsp = human_data['response'][r]
    rn =  human_data['response.number'][r]
    sn = human_data['set.number'][r]
    key = tuple([cl, sn, rn ])
    if rsp == 'F':   human_no[key] += 1
    elif rsp == 'T': human_yes[key] += 1
    elif math.isnan(rsp): continue # a few missing data points
    else:
        assert False, "Error in row %s %s" %(r, rsp)
print "# Loaded human data"

from Model.Data import concept2data
print "# Loaded concept2data"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Inference.GrammarInference.GrammarInference import create_counts

from Model.Grammar import grammar

trees = [h.value for h in hypotheses]

counts, sig2idx, prior_offset = create_counts(grammar, trees, log=LOG)

print "# Computed counts for each hypothesis & nonterminal"

# print counts

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build up the info about the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L        = [] # each hypothesis's cumulative likelihood to each data point
NYes     = []
NTrials  = []
Output  = []
GroupLength = []

for concept in concept2data.keys():

    data = concept2data[concept]

    # compute the likelihood for all the data here
    for h in hypotheses:
        h.stored_likelihood = [h.compute_single_likelihood(d) for d in data]

    for di in xrange(25):
        ll  = [ sum(h.stored_likelihood[:di]) for h in hypotheses ] # each likelihood
        out = [ map(lambda x: 1*h(data[di].input, x), data[di].input) for h in hypotheses] # each response

        # This group will be as long as this number of data points
        GroupLength.append( len(data[di].input) )
        L.append(ll)

        # Each out is a set of responses, that we have to unwrap
        for oi in xrange(len(data[di].input)):
            k = tuple([concept, di+1, oi+1])

            # assert k in human_yes and k in human_no, "*** Hmmm key %s is not in the data" % k
            if k in human_yes or k in human_no:
                Output.append( [x[oi] for x in out] ) # each hypothesis's prediction
                NYes.append( human_yes[k]  )
                NTrials.append( human_no[k] + human_yes[k])
            else:
                print "*** Warning, %s not in human_yes or human_no"%str(k)

    # print "#\t Loaded human data for concept %s" % concept

print "# Created L, NYes, NTrials, and HOutput of size %s" % len(L)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib import break_ctrlc

from LOTlib.Inference.GrammarInference.FullGrammarHypothesis import FullGrammarHypothesis

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

h0 = FullGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
mhs = MHSampler(h0, [], 100000, skip=0)

for s, h in break_ctrlc(enumerate(mhs)):

    print mhs.acceptance_ratio(), h.prior, h.likelihood,\
          h.value['alpha'].value[0], h.value['beta'].value[0],\
          h.value['prior_temperature'].value, h.value['likelihood_temperature'].value,\
          'RULES',\
          ' '.join([str(x) for x in h.value['rulep']['BOOL'].value ]),\
          ' '.join([str(x) for x in h.value['rulep']['PREDICATE'].value ]),\
          ' '.join([str(x) for x in h.value['rulep']['START'].value ]),\
          ' '.join([str(x) for x in h.value['rulep']['SET'].value ])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run gradient ascent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from LOTlib import break_ctrlc
#
# from LOTlib.Inference.GrammarInference.FullGrammarHypothesis import FullGrammarHypothesis
#
#
# from LOTlib.Inference.GrammarInference.GradientGrammarInference import GrammarGradient
#
# h0 = FullGrammarHypothesis(counts, L, GroupLength, prior_offset, NYes, NTrials, Output)
#
# for h in break_ctrlc(GrammarGradient(h0,[])):
#
#     print 0.0, h.prior, h.likelihood,\
#           h.value['alpha'].value[0], h.value['beta'].value[0],\
#           h.value['prior_temperature'].value, h.value['likelihood_temperature'].value,\
#           'RULES',\
#           ' '.join([str(x) for x in h.value['rulep']['BOOL'].value ]),\
#           ' '.join([str(x) for x in h.value['rulep']['PREDICATE'].value ]),\
#           ' '.join([str(x) for x in h.value['rulep']['START'].value ]),\
#           ' '.join([str(x) for x in h.value['rulep']['SET'].value ])
