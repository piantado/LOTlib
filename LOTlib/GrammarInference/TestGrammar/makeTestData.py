"""

Make some test data to try out grammar inference with a known prior

"""
from LOTlib import break_ctrlc

NDATASETS = 50  # how many "Sequences" do we train people on?
DATASET_SIZE = 20 # how long is each sequence?
NPEOPLE = 150
ALPHA = 0.9
BETA  = 0.3 # yes-bias on noise

SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar -- specify a known grammer and see if we can recover its parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

# Set up the grammar
grammar = Grammar()

grammar.add_rule('START', '', ['BOOL'],  0.7)
grammar.add_rule('START', 'True',  None, 0.2)
grammar.add_rule('START', 'False', None, 0.1)

grammar.add_rule('BOOL', 'and_',     ['BOOL', 'BOOL'], 0.1)
grammar.add_rule('BOOL', 'or_',      ['BOOL', 'BOOL'], 0.05)
grammar.add_rule('BOOL', 'not_',     ['BOOL'],         0.025)
grammar.add_rule('BOOL', 'iff_',     ['BOOL', 'BOOL'], 0.0249)
grammar.add_rule('BOOL', 'implies_', ['BOOL', 'BOOL'], 0.0001) # if we sample hypotheses (below), we will have high uncertainty on this
grammar.add_rule('BOOL', '',         ['FEATURE'],      0.8)

grammar.add_rule('FEATURE', 'is_shape_', ['x', 'SHAPE'], 0.3)
grammar.add_rule('FEATURE', 'is_color_', ['x', 'COLOR'], 0.7)

for i, s in enumerate(SHAPES):
    grammar.add_rule('SHAPE', '%s'%q(s), None, 2.0 * (i+1))

for i, c in enumerate(COLORS):
    grammar.add_rule('COLOR', '%s'%q(c), None, 1.0/len(COLORS))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display="lambda x : %s", maxnodes=150, **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

hset = set([make_hypothesis(value=grammar.generate()) for _ in xrange(10000)])
hypotheses = list(hset)

# for h in hypotheses:
    # print h

# hypotheses = []
# for t in grammar.enumerate(d=6):
#     hypotheses.append(make_hypothesis(value=t))

print "# Generated ", len(hypotheses), " hypotheses"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, make_all_objects
from LOTlib.Miscellaneous import sample_one
from random import random

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
        obj = sample_one(all_objects)

        if random() < ALPHA:
            output = target(obj)
        else:
            output = random() < BETA

        data.append( FunctionData(input=[obj], output=output, alpha=ALPHA) )

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

            if random() < ALPHA:
                r =  h(*data[i].input)            # and use it to respond to the next one
            else:
                r = random() < BETA

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
#which_rules = [r for r in grammar if r.nt not in ['START']] # if we want to exclude start
which_rules = [r for r in grammar]

counts, sig2idx, prior_offset = create_counts(grammar, hypotheses, which_rules=which_rules)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Now we save this in hdf5 format that the GPU code can read

import h5py
import numpy
from LOTlib.GrammarInference import export_rule_labels

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

    # these are tab separated since they just get printed by C
    # we must use numpy.string_ to get it in the right format: http://docs.h5py.org/en/latest/strings.html
    # C defaultly reads in 1000 bytes
    #hf.create_dataset('names', data=numpy.string_('\t'.join(r.name for r in which_rules)), dtype="S1000" )
    # hf.create_dataset('names', data='\t'.join(r.name for r in which_rules), dtype=h5py.special_dtype(vlen=bytes) )
    hf.create_dataset('names', data=export_rule_labels(which_rules), dtype=int)
