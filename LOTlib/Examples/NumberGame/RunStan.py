"""
    Convert everything to a stan-runnable file
"""

import pickle
import numpy
from LOTlib.Miscellaneous import qq
from Model import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up some logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import sys

LOG = "stan-log"

if not os.path.exists(LOG):
    os.mkdir(LOG)

# Make sure we print the entire matrix
numpy.set_printoptions(threshold=numpy.inf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

grammar = lot_grammar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses/lot_hypotheses-10.pkl', 'r') as f:
    hypotheses = pickle.load(f)

print "# Loaded hypotheses: ", len(hypotheses)

# - - logging - - - - - - - -
with open(LOG+"/hypotheses.txt", 'w') as f:
    for i, h in enumerate(hypotheses):
        print >>f, i, qq(h)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the concepts from the human data
from Data import load_human_data

human_nyes, human_ntrials = load_human_data()
print "# Loaded human data"

observed_sets = set([ k[0] for k in human_nyes.keys() ])

## TRIM TO FEWER
observed_sets = set(list(observed_sets)[:100])

print "# Loaded %s observed sets" % len(observed_sets)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.GrammarInference.GrammarInference import create_counts

trees = [h.value for h in hypotheses]

which_rules = [ r for r in grammar if r.nt in ['SET', 'MATH', 'EXPR'] ]

counts, sig2idx, prior_offset = create_counts(grammar, trees, which_rules=which_rules)

print "# Computed counts for each hypothesis & nonterminal"

# - - logging - - - - - - - -
for nt in counts.keys():
    with open(LOG+"/counts_%s.txt"%nt, 'w') as f:
        for r in xrange(counts[nt].shape[0]):
            print >>f, r, ' '.join(map(str, counts[nt][r,:].tolist()))

# - - logging - - - - - - - -
with open(LOG+"/sig2idx.txt", 'w') as f:
    for s in sorted(sig2idx.keys(), key=lambda x: (x[0], sig2idx[x]) ):
        print >>f,  s[0], sig2idx[s], qq(s)

# - - logging - - - - - - - -
with open(LOG+"/prior_offset.txt", 'w') as f:
    for i, x in enumerate(prior_offset):
        print >>f, i, x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build up the info about the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData

L           = [] # each hypothesis's cumulative likelihood to each data point
GroupLength = []
NYes        = []
NTrials     = []
Output     = []

domain = range(1,101)

for os in observed_sets:

    datum = FunctionData(input=[], output=os, alpha=ALPHA)

    # compute the likelihood for all the data here
    for h in hypotheses:
        h.cached_set = h()
        h.stored_likelihood = h.compute_single_likelihood(datum, cached_set=h.cached_set)

    L.append([ h.stored_likelihood for h in hypotheses ]) # each likelihood
    GroupLength.append(len(domain))

    for i in domain:
        Output.append( [ 1*(i in h.cached_set) for h in hypotheses])
        NYes.append( human_nyes[tuple([os,i])] )
        NTrials.append( human_ntrials[tuple([os,i])] )

    # print "#\t Loaded observed set %s" % str(os)

print "# Created NYes, NTrials, and Output of size %s" % len(L)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Stan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pystan
from LOTlib.GrammarInference.GrammarInference import make_stan_code

stan_code = make_stan_code(counts)

stan_data = {
    'N_HYPOTHESES': len(hypotheses),
    'N_DATA': len(NTrials),
    'N_GROUPS': len(GroupLength),

    'PriorOffset': prior_offset,

    'L': L,
    'GroupLength':GroupLength,

    'NYes':    NYes,
    'NTrials': NTrials,
    'Output': Output
}
stan_data.update({ 'count_%s'%nt:counts[nt] for nt in counts.keys()}) # add the prior counts. Note we have to convert their names here

print "# Summary of model size:"
for nt in counts:
    print "# Matrix %s is %s x %s" % (nt, counts[nt].shape[0], counts[nt].shape[1])


# - - logging - - - - - - - -
with open(LOG+"/model.stan", 'w') as f:
    print >>f, stan_code

print "# Running"
fit = pystan.stan(model_code=stan_code,  data=stan_data, warmup=50, iter=500, chains=4)
print(fit)

## And save
# with open("stan_fit.pkl", 'w') as f:
#     pickle.dump(fit, f)

# fit.plot().savefig("fig.pdf")

# print(fit.extract(permuted=True))