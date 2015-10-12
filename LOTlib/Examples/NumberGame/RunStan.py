"""
    Convert everything to a stan-runnable file
"""

import pickle
import numpy
from LOTlib.Miscellaneous import qq, setup_directory
from Model import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up some logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOG = "stan-log"
setup_directory(LOG) # make a directory for ourselves

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
# observed_sets = set(list(observed_sets)[:100])

print "# Loaded %s observed sets" % len(observed_sets)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.GrammarInference.GrammarInference import create_counts

trees = [h.value for h in hypotheses]

# Decide which rules to use
which_rules = [ r for r in grammar if r.nt in ['SET', 'MATH', 'EXPR'] ]

counts, sig2idx, prior_offset = create_counts(grammar, trees, which_rules=which_rules, log=LOG)

print "# Computed counts for each hypothesis & nonterminal"


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

    gl = 0 # how many did we actually ad?
    for i in domain:
        k = tuple([os,i])

        if k in human_nyes and k in human_ntrials:
            gl += 1
            Output.append( [ 1*(i in h.cached_set) for h in hypotheses])
            NYes.append( human_nyes[k] )
            NTrials.append( human_ntrials[k] )
        else:
            print "*** Warning %s not in human data!" % str(k)

    GroupLength.append(gl)

    # print "#\t Loaded observed set %s" % str(os)

print "# Created NYes, NTrials, and Output of size %s" % len(L)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Stan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pystan
from LOTlib.GrammarInference.GrammarInference import make_stan_code

stan_code = make_stan_code(counts, log=LOG)

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

sm = pystan.StanModel(model_code=stan_code)

print "# Created Stan model"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Stan optimization and bootstrap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fit = sm.optimizing(data=stan_data)

for k in fit:
    if isinstance(fit[k].tolist(), list):
        for i, v in enumerate(fit[k].tolist()):
            print "real", k, i, v
    else:
        print "real", k, 0, fit[k].tolist()


## And bootstrap
import scipy.stats

y = numpy.array(NYes, dtype=float) # so we can take ratios
n = numpy.array(NTrials)

for rep in xrange(100):

    # Resample our yeses
    stan_data['NYes'] = scipy.stats.binom.rvs(n, y/n)

    # and re-run
    fit = sm.optimizing(data=stan_data)

    for k in fit:
        if isinstance(fit[k].tolist(), list):
            for i, v in enumerate(fit[k].tolist()):
                print "boot", k, i, v
        else:
            print "boot", k, 0, fit[k].tolist()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Stan sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# samples = sm.sampling(data=stan_data, iter=100, chains=4, sample_file="./stan_samples")

# print "# Running"
# fit = pystan.stan(model_code=stan_code,  data=stan_data, warmup=50, iter=500, chains=4)
# print(fit)

## And save
# with open("stan_fit.pkl", 'w') as f:
#     pickle.dump(fit, f)

# fit.plot().savefig("fig.pdf")

# print(fit.extract(permuted=True))