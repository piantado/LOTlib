"""
    Convert everything to a stan-runnable file
"""

import pickle

ALPHA = 0.99

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model.Grammar import lot_grammar as grammar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses.pkl', 'r') as f:
    hypotheses = pickle.load(f)

print "# Loaded hypotheses: ", len(hypotheses)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the concepts from the human data
from Data import load_human_data

human_nyes, human_ntrials = load_human_data()
print "# Loaded human data"

observed_sets = set([ k[0] for k in human_nyes.keys() ])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.GrammarInference.GrammarInference import create_counts

trees = [h.value for h in hypotheses]

counts, sig2idx = create_counts(grammar, trees)

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
    GroupLength.append(len(domain))

    for i in domain:
        Output.append( [ 1*(i in h.cached_set) for h in hypotheses])
        NYes.append( human_nyes[tuple([os,i])] )
        NTrials.append( human_ntrials[tuple([os,i])] )

    print "#\t Loaded observed set %s" % str(os)

print "# Created NYes, NTrials, and Output of size %s" % len(L)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Stan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pystan
from LOTlib.GrammarInference.GrammarInference import make_stan_code

rule_counts = {nt: counts['count_%s'%nt].shape[1] for nt in grammar.nonterminals()}
stan_code = make_stan_code(rule_counts)

stan_data = {
    'N_HYPOTHESES': len(hypotheses),
    'N_DATA': len(NTrials),
    'N_GROUPS': len(GroupLength),

    'L': L,
    'GroupLength':GroupLength,

    'NYes':    NYes,
    'NTrials': NTrials,
    'Output': Output
}
stan_data.update(counts) # and the prior counts

for s in sorted(sig2idx.keys(), key=lambda x: (x[0], sig2idx[x]) ):
    print s, sig2idx[s]

print "# Saving stan data"

with open("stan_data.pkl", 'w') as f:
    pickle.dump(stan_data, f)


print "# Running with code\n", stan_code

fit = pystan.stan(model_code=stan_code,  data=stan_data, iter=150, chains=4)
print(fit)

fit.plot().savefig("fig.pdf")


## And save
with open("stan_fit.pkl", 'w') as f:
    pickle.dump(fit, f)


# print(fit.extract(permuted=True))