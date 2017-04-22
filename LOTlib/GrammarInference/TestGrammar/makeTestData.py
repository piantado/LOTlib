"""

Make some test data to try out grammar inference with a known prior

"""
from LOTlib import break_ctrlc

NDATASETS = 30  # how many "Sequences" do we train people on?
DATASET_SIZE = 10 # how long is each sequence?
NPEOPLE = 50

SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar -- specify a known grammer and see if we can recover its parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

# Set up the grammar
# Here, we create our own instead of using DefaultGrammars.Nand because
# we don't want a BOOL/PREDICATE distinction
grammar = Grammar()

grammar.add_rule('START', '', ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'],  5.)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'],   1.)
grammar.add_rule('BOOL', 'not_', ['BOOL'],          2.)

# And finally, add the primitives
for s in SHAPES:
    grammar.add_rule('BOOL', 'is_shape_', ['x', q(s)], 3.)

for c in COLORS:
    grammar.add_rule('BOOL', 'is_color_', ['x', q(c)], 5.)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display="lambda x : %s", **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

hypotheses = []
for t in grammar.enumerate(d=4):
    hypotheses.append(make_hypothesis(value=t))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, make_all_objects
from LOTlib.Miscellaneous import sample_one

all_objects = make_all_objects( shape=SHAPES, color=COLORS )

# Generate all of the data, picking a different target hypothesis for each sequence
# "people" are shown
datas = []
for di in xrange(NDATASETS):
    # pick a hypothesis at random
    target = sample_one(hypotheses)
    print "# Target:", target
    data = []
    for _ in xrange(DATASET_SIZE):
        o = sample_one(all_objects)
        data.append( FunctionData(input=[o], output=target(o), alpha=0.90) )

    datas.append(data)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulate people's response
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Miscellaneous import weighted_sample

NYes = [0] * (DATASET_SIZE*NDATASETS) #number of yes/no responses for each
NNo  = [0] * (DATASET_SIZE*NDATASETS)

di = 0
for datasi, data in enumerate(datas):
    print "# Simulating data for ", datasi
    for i in xrange(len(data)):

        # update the posterior
        for h in hypotheses:
            h.compute_posterior( [data[j] for j in xrange(i)])
        probs = [x.posterior_score for x in hypotheses]
        # sample (if this is the hypothesis)
        for person in break_ctrlc(xrange(NPEOPLE)):
            h = weighted_sample(hypotheses, probs=probs, log=True)
            r = h(*data[i].input)            # and use it to respond to the next one
            if r: NYes[di] += 1
            else: NNo[di]  += 1

        di += 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Take into account the likelihoods in our inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# L is in order l1h1, l1h2... l2h1, l2h2...
# this is because the GPU's compute_human_likelihood parallelizes this
# with each thread looping over h_i and it is most efficient if threads
# access adjacent memory locations
L = []
output = []

for h in hypotheses:

    for data in datas:

        # get each hypothesis' response to each data point
        # Note that since we model a response after data[0] AND data[1], we must include_last
        predll = h.compute_predictive_likelihood(data, include_last=True)

        for i in xrange(len(data)):
            output.append( 1.0 * h(*data[i].input) ) # give this hypothesis' output to the object
            L.append(predll[i])                 # what was the likelihood on all previous data for that output?

assert len(L) == len(output)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the prior counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.GrammarInference import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt not in ['START']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Now we save this in hdf5 format that the GPU code can read

import h5py
import numpy

# stack together counts
kys = counts.keys()
ntlen = [len(counts[k][0]) for k in kys] # store how long each
counts = numpy.concatenate([counts[k] for k in kys], axis=1)
NHyp = len(hypotheses)

print "const int NRULES = %s;" % sum(ntlen)
print "const int NHYP = %s;" % NHyp
print "const int NDATA = %s;" % len(NYes)
print "const int NNT = %s;" % len(ntlen)
print "const int NTLEN[NNT] = {%s};" % ','.join(map(str,ntlen))

# print out the key for understanding the columns
sigs = sig2idx.keys()
for k in kys:
    s = [sig for sig in sigs if sig[0] == k]
    s = sorted(s, key=sig2idx.get)
    for si in s:
        print sig2idx[si], si

# export as hdf5
with h5py.File('data.h5', 'w') as hf:
    # first write the 'specs'
    hf.create_dataset('specs',    data=[NHyp, sum(ntlen), len(NYes), len(ntlen)], dtype=int)
    # write all the data
    hf.create_dataset('counts',    data=numpy.ravel(counts[:NHyp ], order='C'), dtype=int) # must be sure if we stop early, we don't include too many counts
    hf.create_dataset('output',    data=output, dtype=float)
    hf.create_dataset('human_yes', data=numpy.ravel(NYes, order='C'), dtype=int)
    hf.create_dataset('human_no',  data=numpy.ravel(NNo, order='C'), dtype=int)
    hf.create_dataset('likelihood',  data=numpy.ravel(L, order='C'), dtype=float)
    hf.create_dataset('ntlen', data=ntlen, dtype=int)
