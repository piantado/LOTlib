"""
    We'll model an "experiment" (simulated data) where subjects give there responses to *all*
    objects in the collection: first, with no data, then after observing data[0], then
    after observing data[1] too.

"""

from Model import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For now, we'll just sample from the prior
hypotheses = list(set([MyHypothesis(grammar=grammar, maxnodes=100) for _ in xrange(1000)])) # list so order is maintained

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This stores each hypothesis vs a vector of counts of how often each nonterminal is used
# this is used via a matrix product with the log priors on the GPU to compute the prior
# (the (log)priors are the things we are trying to infer)

from LOTlib.GrammarInference.Precompute import create_counts

# Decide which rules to use
which_rules = [r for r in grammar if r.nt not in ['START']]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)

print "# Computed counts for each hypothesis & nonterminal"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import make_all_objects

objects = make_all_objects(size=['small', 'medium', 'large'],
                           color=['red', 'green', 'blue'],
                           shape=['square', 'triangle', 'circle'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The data that learners observed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = [FunctionData(input=[Obj(size='small', color='green', shape='square')], output=True, alpha=0.99),
        FunctionData(input=[Obj(size='large', color='red', shape='triangle')], output=False, alpha=0.99)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Human data, we'll simulate on all objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NYes = []
NNo = []

for r in xrange(len(data)+1): # Each response, one with no data, then conditioned on each data[i]

    # If we wanted to simulate people changing beliefs, we could enter it here
    # as a function of r. For now, we will just assume they always give the same answer

    for o in objects:

        # Here, in this simulated data, is where we simulate what people have in mind.
        # They have observed data INCONSISTENT red,square, so red responses should make
        # you think the prior on red is very high
        if o.color=='red': # Simulate people who have in mind the concept "red"
            NYes.append(10)
            NNo.append(0)
        else:
            NYes.append(0)
            NNo.append(10)

    assert len(NYes) == len(NNo)

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

    # get each hypothesis' response to each data point
    # Note that since we model a response after data[0] AND data[1], we must include_last
    predll = h.compute_predictive_likelihood(data, include_last=True)

    for i in xrange(len(data)):
        for o in objects:
            output.append( 1.0 * h(o) ) # give this hypothesis' output to the object
            L.append(predll[i])             # what was the likelihood on all previous data for that output?

    assert len(L) == len(output)


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
