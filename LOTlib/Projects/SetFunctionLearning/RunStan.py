"""
    Convert everything to a stan-runnable file
"""
import pickle
import numpy

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up some logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Miscellaneous import setup_directory, qq

LOG = "stan-log"
setup_directory(LOG) # make a directory for ourselves

# Make sure we print the entire matrix
numpy.set_printoptions(threshold=numpy.inf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses/hypotheses-10.pkl', 'r') as f:
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
# Set up stan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pystan
from LOTlib.Inference.GrammarInference.GrammarInference import make_stan_code

stan_code = make_stan_code(counts)

stan_data = {
    'N_HYPOTHESES': len(hypotheses),
    'N_DATA': len(NTrials),
    'N_GROUPS': len(GroupLength),

    'PriorOffset': prior_offset,

    'L': L,
    'GroupLength': GroupLength,

    'NYes':    NYes,
    'NTrials': NTrials,
    'Output': Output
}
stan_data.update({ 'count_%s'%nt:counts[nt] for nt in counts.keys()}) # add the prior counts. Note we have to convert their names here

print "# Summary of model size:"
for nt in counts:
    print "# Matrix %s is %s x %s" % (nt, counts[nt].shape[0], counts[nt].shape[1])


model_code = make_stan_code(counts, log=LOG)

sm = pystan.StanModel(model_code=model_code)

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
# Run Stan sampling -- very slow
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
