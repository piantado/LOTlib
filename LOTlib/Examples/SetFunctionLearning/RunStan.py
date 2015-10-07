"""
    Convert everything to a stan-runnable file
"""
import pickle

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses.pkl', 'r') as f:
    concept2hypotheses = pickle.load(f)
print "# Loaded hypotheses: ", map(len, concept2hypotheses.values())

hypotheses = set()
for hset in concept2hypotheses.values():
    hypotheses.update(hset)

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

from LOTlib.GrammarInference.GrammarInference import create_counts

from Model.Grammar import grammar

trees = [h.value for h in hypotheses]

counts, sig2idx, prior_offset = create_counts(grammar, trees)

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
        out = [ map(lambda x: 1*h(x), data[di].input) for h in hypotheses] # each response

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

    print "#\t Loaded concept %s"%concept

print "# Created L, NYes, NTrials, and HOutput of size %s" % len(L)

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
    'GroupLength': GroupLength,

    'NYes':    NYes,
    'NTrials': NTrials,
    'Output': Output
}
stan_data.update({ 'count_%s'%nt:counts[nt] for nt in counts.keys()}) # add the prior counts. Note we have to convert their names here

for s in sorted(sig2idx.keys(), key=lambda x: (x[0], sig2idx[x]) ):
    print s, sig2idx[s]

print "# Saving stan data"

# with open("stan_data.pkl", 'w') as f:
#     pickle.dump(stan_data, f)


print "# Running with code\n", stan_code

fit = pystan.stan(model_code=stan_code,  data=stan_data, iter=50, chains=2)
fit.plot().savefig("fig.pdf")
print(fit)

## And save
with open("stan_fit.pkl", 'w') as f:
    pickle.dump(fit, f)


# print(fit.extract(permuted=True))